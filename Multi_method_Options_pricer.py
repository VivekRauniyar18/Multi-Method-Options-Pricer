"""
Multi-method Options Pricer
1. Black-Scholes(closed-form, European)
2. Black-Scholes-Merton (closed-form with continuous dividend yield)
3. Cox-Ross-Rubinstein binomial tree( European)
4. Monte Carlo Simulation (European)
5. Monte Carlo Simulation (Asian- average-price)
6. Monte Carlo Simulation (Barrier-up-and-in)

"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable
from scipy.stats import norm

# Data model
@dataclass(frozen=True)
class OptionContract:
    """An option contract on a single underlying."""
    ticker: str
    spot: float                     # S - current price of underlying
    strike: float                   # K - strike price
    maturity: float                 # T - time to expiry, in years
    rate: float                     # r - risk-free rate, annual, continuously compound
    volatility: float               # sigma - annualize volatility
    dividend_yield: float = 0.0           # q - continuous dividend yield

    def describe(self) -> str:
        return(
            f"{self.ticker} S={self.spot} K={self.strike} "
            f"T={self.maturity:.4f}y  r={self.rate:.2%} "
            f"sigma={self.volatility:.2%} q={self.dividend_yield:.2%}"
        )

@dataclass
class PricingResult:
    """Result of a pricing call. Holds the price plus optional Monte Carlo error."""
    method: str
    call: float
    put: float
    call_std_error: float | None = None
    put_std_error: float | None = None   



# 1. Black-Scholes / Black-Scholes-Merton (closed-form)

class BlackScholes:
    """Closed-form European option pricing.
    The Black-Scholes-Merton extensions supports a continuous dividend yield q:
    pass q = 0 to recover plain Black-Scholes. """

    @staticmethod
    def _d1_d2(c: OptionContract) -> tuple[float, float]:
        S, K, T, r, sigma, q = (c.spot, c.strike, c.maturity, c.rate, 
                                c.volatility, c.dividend_yield)
        
        d1 = (np.log(S / K) + (r-q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    

    @staticmethod
    def price(c: OptionContract) -> PricingResult:
        d1, d2 = BlackScholes._d1_d2(c)
        disc_r = np.exp(-c.rate * c.maturity)
        disc_q = np.exp(-c.dividend_yield * c.maturity)
        call = c.spot * disc_q * norm.cdf(d1) - c.strike * disc_r * norm.cdf(d2)
        put = c.strike * disc_r * norm.cdf(-d2) - c.spot * disc_q * norm.cdf(-d1)
        label = "Black-Scholes" if c.dividend_yield ==0 else "Black-Scholes-Merton"
        return PricingResult(method=label, call= call, put=put)
    

    @staticmethod
    def greeks(c: OptionContract) -> dict:
        """Closed-form Greeks. Useful for delta-hedging and risk reports."""
        d1, d2 = BlackScholes._d1_d2(c)
        S, K, T, r, sigma, q  = (c.spot, c.strike, c.maturity, c.rate, c.volatility, c.dividend_yield)
        disc_r = np.exp(-r * T)
        disc_q = np.exp(-q *T)
        pdf_d1 = norm.pdf(d1)


        return {
            "delta_call": disc_q *norm.cdf(d1),
            "delta_put": -disc_q * norm.cdf(-d1),
            "gamma": disc_q * pdf_d1 / (S * sigma * np.sqrt(T)),
            "vega": S * disc_q * pdf_d1 * np.sqrt(T), # per 1.00 vol change
            "theta_call": -(S * disc_q * pdf_d1 * sigma) / (2 * np.sqrt(T))
                          - r * K * disc_r * norm.cdf(d2)  
                          + q * S * disc_q * norm.cdf(d1),
            "rho_call": K * T * disc_r * norm.cdf(d2),
            "rho_put": -K * T * disc_r * norm.cdf(-d2),  

        }


# 2. Cox-Ross-Rubinstein binomial tree

class BinomialTree:
    """ Cox-Ross-Rubinstein binomial tree for European options.

    it uses the standard parametrization:
    u = exp(sigma root under del t), d = 1/u
    p = (exp((r-q) del t) - d) / (u - d)

    Backward-induction is performed on level-by-level arrays.
    """

    @staticmethod
    def price(c: OptionContract, steps: int= 500) -> PricingResult:
        dt = c.maturity / steps 
        u = np.exp(c.volatility * np.sqrt(dt))
        d = 1.0/u
        p = (np.exp((c.rate - c.dividend_yield) * dt) -d) / (u-d)
        disc = np.exp(-c.rate * dt)

        # Terminal stock prices: S * u^(steps -i) * d^i for i in 0.. steps
        i = np.arange(steps + 1)
        S_T = c.spot * (u ** (steps-i)) * (d**i)

        #Terminal payoffs
        call_vals = np.maximum(S_T - c.strike, 0.0)
        put_vals = np.maximum(c.strike - S_T, 0.0)

        #Backward Induction
        for _ in range(steps):
            call_vals = disc * (p * call_vals[:-1] + (1-p) * call_vals[1:])
            put_vals = disc * (p * put_vals[:-1] + (1-p) * put_vals[1:])

        return PricingResult(method=f"Binomial Tree ({steps} steps)",
                            call = float(call_vals[0]),
                            put = float(put_vals[0]))


# Monte Carlo simulation engine

class MonteCarloEngine:
    """Monte Carlo pricing under risk-neutral geometric Brownian motion."""

    def __init__(self, n_paths:int = 10000, n_steps: int = 252,
                 seed: int | None = 42, antithetic: bool = True):
        
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.antithetic = antithetic
        self.rng = np.random.default_rng(seed)


       
# internal: simulate the full price grid(paths x time)

    def _simulate_paths(self, c: OptionContract) -> np.ndarray:
        dt = c.maturity / self.n_steps
        drift = (c.rate - c.dividend_yield - 0.5 * c.volatility **2) * dt
        diffuse = c.volatility * np.sqrt(dt)


        # Antithetic variates : halve sampling variance for free
        if self.antithetic:
            half = self.n_paths //2
            Z_half = self.rng.standard_normal((half, self.n_steps))
            Z = np.concatenate([Z_half, -Z_half], axis = 0)
        else:
            Z = self.rng.standard_normal((self.n_paths, self.n_steps))

        log_increments = drift + diffuse * Z
        log_paths = np.cumsum(log_increments, axis = 1)
        paths = c.spot * np.exp(log_paths)
        paths = np.column_stack([np.full(paths.shape[0], c.spot), paths])
        return paths # shape (n_paths,  n_steps + 1)


# European: terminal payoff pay only
    def price_european(self, c:OptionContract) -> PricingResult:
        paths = self._simulate_paths(c)
        S_T = paths[:, -1]
        call_payoffs = np.maximum(S_T - c.strike, 0.0)
        put_payoffs = np.maximum(c.strike - S_T, 0.0)
        disc = np.exp(-c.rate * c.maturity)

        return PricingResult(
            method = f"Monte Carlo - European(N ={self.n_paths:,})",
            call = disc * call_payoffs.mean(),
            put = disc * put_payoffs.mean(),
            call_std_error= disc * call_payoffs.std(ddof=1) / np.sqrt(self.n_paths),
            put_std_error= disc * put_payoffs.std(ddof = 1) / np.sqrt(self.n_paths),
        )


    # Asian: avearage-price payoff
    def price_asian(self, c:OptionContract,
                    average_kind : str = "arithmetic") -> PricingResult:
        paths = self._simulate_paths(c)
        if average_kind=="arithmetic":
            S_avg = paths[:, 1:].mean(axis = 1)
        elif average_kind=="geometric":
            S_avg = np.exp(np.log(paths[:,1:]).mean(axis=1))
        else:
            raise ValueError("Average_kind must be 'arithmetic' or 'geometric'")

        call_payoffs= np.maximum(S_avg - c.strike, 0.0)
        put_payoffs = np.maximum(c.strike - S_avg, 0.0)
        disc = np.exp(-c.rate * c.maturity)

        return PricingResult(
            method = f"Monte Carlo - Asian ({average_kind})",
            call = disc * call_payoffs.mean(),
            put = disc * put_payoffs.mean(),
            call_std_error= disc * call_payoffs.std(ddof =1) / np.sqrt(self.n_paths),
            put_std_error= disc * put_payoffs.std(ddof=1) / np.sqrt(self.n_paths),
        )

 # Barrier = up-and -in (most common knock-in variant)
    def price_barrier(self, c:OptionContract, barrier = float,
                      kind: str= "up-and-in") -> PricingResult:

    # Knock-in/knock-out barrier option.
    # Supported kinds: 'up-and-in', 'up-and-out', 'down-and-in', 'down-and-out'

        paths = self._simulate_paths(c)
        S_T = paths[:, -1]
        path_max = paths.max(axis=1)
        path_min = paths.min(axis=1)

        if kind == "up-and-in": active = path_max >= barrier
        elif kind == "up-and-out": active = path_max < barrier
        elif kind == "down-and-in": active = path_min <= barrier
        elif kind == "down-and-out": active = path_min > barrier
        else:
            raise ValueError(f"Unknown barrier kind: {kind}")

        call_payoffs = np.where(active, np.maximum(S_T - c.strike, 0.0), 0.0)
        put_payoffs = np.where(active, np.maximum(c.strike  - S_T, 0.0), 0.0)
        disc = np.exp(-c.rate * c.maturity)

        return PricingResult(
            method = f"Monte Carlo - Barrier ({kind}, B = {barrier})",
            call  = disc * call_payoffs.mean(),
            put = disc * put_payoffs.mean(),
            call_std_error= disc * call_payoffs.std(ddof=1) / np.sqrt(self.n_paths),
            put_std_error= disc * put_payoffs.std(ddof=1) / np.sqrt(self.n_paths),
        )


     # Historical volatility estimation

def Historical_volatility(prices: Iterable[float],trading_days: int = 252) -> tuple[float, float]:


    #Estimate annulaixse volatility from a series of close prices.
    # Returns (sigma_annual, sigma_daily). Use log returns rather than simple returns
    # (the standard convention for Black-Scholes calibration).

    p = np.asarray(prices, dtype=float)
    log_retuns = np.diff(np.log(p))
    sigma_daily = log_retuns.std(ddof=1)
    sigma_annual = sigma_daily * np.sqrt(trading_days)
    return sigma_annual, sigma_daily


    

 # Reporting 

def format_results(results: list[PricingResult]) -> pd.DataFrame:
        
        """Building a tidy comparison table from a liost of PricingResults."""
        rows = []
        for r in results:
            call_err = f"+- {r.call_std_error:.4f}" if r.call_std_error else""
            put_err = f" +-{r.put_std_error:.4f}" if r.put_std_error else""
            rows.append({
                "Method": r.method,
                "Call": f"{r.call:8.4f}{call_err}",
                "Put":  f"{r.put:8.4f}{put_err}",
                })
        return pd.DataFrame(rows)
    

# Main demo: reproduces the SPY example from the original workbook

def main(csv_path: str = "spy_data.csv"):
        print("=" * 78)
        print("MULTI-METHOD OPTIONS PRICER - SPY EXAMPLE") 
        print("=" * 78)

        # Calibrate volatility fro historical prices

        spy = pd.read_csv(csv_path)
        sigma_annual, sigma_daily = Historical_volatility(spy["Adj Close"])
        print(f"\nHistorical SPY data: {spy['Date'].iloc[0]} to {spy['Date'].iloc[-1]}")
        print(f" Daily log-return volatility : {sigma_daily:.4%}")
        print(f" Annulaized volatility (sigma * root under 252): {sigma_annual:.4%}")


        # Define the contract- SPY 555 call/put, 6 trading days to expiry

        contract = OptionContract(
            ticker = "SPY",
            spot = 549.0,
            strike= 555.0,
            maturity=6/252,
            rate = 0.055,
            volatility=sigma_annual,
            dividend_yield=0.0,
        )
        print(f"\nContract: {contract.describe()}\n")

        # price using all available mathods 
        results: list[PricingResult] = []

        results.append(BlackScholes.price(contract))

        # BSM variant with continuous divident yield estimate (trailing 4Q= 1.25%)

        contract_with_div = OptionContract(
            **{**contract.__dict__, "dividend_yield": 0.01248}
        )
        results.append(BlackScholes.price(contract_with_div))

        results.append(BinomialTree.price(contract, steps=500))

        mc = MonteCarloEngine(n_paths=100000, n_steps=252, seed=42, antithetic=True)
        results.append(mc.price_european(contract))
        results.append(mc.price_barrier(contract, barrier=557.0, kind="up-and-in"))


        # Display
        df = format_results(results)
        print(df.to_string(index=False))

        # Greeks
        print("\n -- Black-scholes Greeks (no dividend) --")
        greeks = BlackScholes.greeks(contract)
        for k, v in greeks.items():
            print(f" {k:12s}: {v:+.6f}")

        print("\nDone.")

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="spy_data.csv",
                        help="path to historical price CSV (needs 'Adj Close')")
    
    args = parser.parse_args()
    main(args.csv)


        
