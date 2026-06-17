from math import erf, floor, sqrt

import numpy as np
import pandas as pd


def american_to_decimal(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def decimal_to_american(decimal_odds: float) -> float:
    if pd.isna(decimal_odds) or decimal_odds <= 1:
        return np.nan
    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100, 0)
    return round(-100 / (decimal_odds - 1), 0)


def normal_cdf(x: float, mean: float, std: float) -> float:
    std = max(float(std), 1e-6)
    z = (x - mean) / (std * sqrt(2))
    return 0.5 * (1 + erf(z))


def over_probability(projection: float, line: float, residual_std: float) -> float:
    return 1 - normal_cdf(line, projection, residual_std)


def poisson_over_probability(lam: float, line: float) -> float:
    """P(X > line) where X ~ Poisson(lam).

    For a half-integer line like 5.5: P(X > 5.5) = P(X >= 6) = 1 - P(X <= 5).
    For an integer line like 6: P(X > 6) = P(X >= 7) = 1 - P(X <= 6).
    scipy.stats.poisson.sf(k, mu) = P(X > k), so sf(floor(line), lam) handles both.
    """
    if pd.isna(lam) or pd.isna(line):
        return np.nan
    lam = max(float(lam), 1e-6)
    k = int(floor(float(line)))
    try:
        from scipy.stats import poisson as _poisson
        return float(_poisson.sf(k, lam))
    except ImportError:
        # Fallback: normal approximation using Poisson std = sqrt(lambda)
        return over_probability(lam, line + 0.5, sqrt(lam))


def nb_over_probability(mu: float, alpha: float, line: float) -> float:
    """P(X > line) where X ~ NegativeBinomial(mean=mu, dispersion=alpha).

    Var(X) = mu + alpha*mu^2. When alpha approaches 0 this converges to Poisson.
    scipy nbinom parameterisation: n=1/alpha, p=n/(n+mu).
    Falls back to Poisson when alpha <= 0 or unavailable.
    """
    if pd.isna(mu) or pd.isna(line) or pd.isna(alpha) or alpha <= 0:
        return poisson_over_probability(mu, line)
    mu = max(float(mu), 1e-6)
    alpha = max(float(alpha), 1e-9)
    k = int(floor(float(line)))
    try:
        from scipy.stats import nbinom
        n = 1.0 / alpha
        p = n / (n + mu)
        return float(nbinom.sf(k, n, p))
    except ImportError:
        return poisson_over_probability(mu, line)


def shrink_probability(probability: float, shrink_factor: float) -> float:
    if pd.isna(probability):
        return np.nan
    shrink_factor = min(max(float(shrink_factor), 0), 1)
    return 0.5 + (float(probability) - 0.5) * shrink_factor


def fair_american_odds(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1 - 1e-6)
    return decimal_to_american(1 / probability)


def expected_value(probability: float, american_odds: float) -> float:
    decimal_odds = american_to_decimal(american_odds)
    if pd.isna(decimal_odds):
        return np.nan
    profit = decimal_odds - 1
    return probability * profit - (1 - probability)


def kelly_fraction(probability: float, american_odds: float, max_fraction: float) -> float:
    decimal_odds = american_to_decimal(american_odds)
    if pd.isna(decimal_odds):
        return np.nan
    b = decimal_odds - 1
    q = 1 - probability
    raw = (b * probability - q) / b
    return float(min(max(raw, 0), max_fraction))


def add_betting_columns(
    df: pd.DataFrame,
    market: str,
    residual_std: float,
    max_kelly_fraction: float,
    edge_shrink_factor: float = 1.0,
    distribution: str = "normal",
    bias_correction: float = 0.0,
    nb_alpha=None,
) -> pd.DataFrame:
    """Compute over/under probabilities, fair odds, EV, and Kelly fraction.

    distribution:    "normal"           uses Gaussian with residual_std.
                     "poisson"          uses the Poisson CDF.
                     "negative_binomial" uses NB CDF (nb_alpha required).
    bias_correction: added to projection before computing probability.
    nb_alpha:        NB dispersion parameter (Var = mu + alpha*mu^2).
                     Only used when distribution="negative_binomial".
    """
    out = df.copy()
    projection_col = f"{market}_projection"

    if distribution == "negative_binomial" and nb_alpha is not None:
        out["raw_over_probability"] = out.apply(
            lambda row: nb_over_probability(
                row[projection_col] + bias_correction, nb_alpha, row["line"]
            ),
            axis=1,
        )
    elif distribution == "poisson":
        out["raw_over_probability"] = out.apply(
            lambda row: poisson_over_probability(
                row[projection_col] + bias_correction, row["line"]
            ),
            axis=1,
        )
    else:
        out["raw_over_probability"] = out.apply(
            lambda row: over_probability(
                row[projection_col] + bias_correction, row["line"], residual_std
            ),
            axis=1,
        )

    out["over_probability"] = out["raw_over_probability"].map(
        lambda p: shrink_probability(p, edge_shrink_factor)
    )
    out["under_probability"] = 1 - out["over_probability"]
    out["fair_over_odds"] = out["over_probability"].map(fair_american_odds)
    out["fair_under_odds"] = out["under_probability"].map(fair_american_odds)
    out["over_ev"] = out.apply(
        lambda row: expected_value(row["over_probability"], row.get("over_odds", np.nan)),
        axis=1,
    )
    out["under_ev"] = out.apply(
        lambda row: expected_value(row["under_probability"], row.get("under_odds", np.nan)),
        axis=1,
    )
    out["best_side"] = np.where(out["over_ev"] >= out["under_ev"], "over", "under")
    out["ev"] = np.where(out["best_side"] == "over", out["over_ev"], out["under_ev"])
    out["edge_pct"] = out["ev"] * 100
    out["kelly_fraction"] = out.apply(
        lambda row: kelly_fraction(
            row["over_probability"] if row["best_side"] == "over" else row["under_probability"],
            row["over_odds"] if row["best_side"] == "over" else row["under_odds"],
            max_kelly_fraction,
        ),
        axis=1,
    )
    return out
