"""Correlation and goodness-of-fit statistics for training curves.

Two questions come up when reading a run, and they need different statistics:

**Is the run converging cleanly?**
    Spearman's rho against epoch. It is a *rank* correlation, so it measures
    monotonicity without assuming the curve has any particular shape. A loss
    that falls steeply then plateaus is still perfectly monotone; rho catches
    that where a straight-line fit would not.

**Is the objective a good proxy for the thing I actually care about?**
    Spearman's rho between the loss and the physical figure of merit. The loss
    here is a log intensity ratio and the figure of merit is contrast in dB.
    Nothing guarantees a priori that minimising one maximises the other, and if
    they decouple, the run is optimising the wrong thing.

R-squared is reported alongside, always for an ordinary least-squares *straight
line* -- which for a simple linear fit is exactly Pearson r squared. Quoting it
next to rho is deliberate: where rho is near +/-1 and R-squared is well below
it, the relationship is monotone but curved, which is the normal signature of a
converging loss and not a problem.

Implemented in numpy rather than pulled from scipy so the statistics stay
inspectable and the package keeps one fewer hard dependency.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "rank_average",
    "pearson",
    "spearman",
    "r_squared",
    "linear_fit",
    "running_spearman",
    "correlation_report",
]


def rank_average(x) -> np.ndarray:
    """Ranks of ``x``, with ties sharing their average rank.

    Average ranks (rather than ordinal ones) are what make Spearman's rho
    well defined when values repeat -- and training curves do repeat, whenever
    an epoch fails to move a metric.
    """
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)

    # average the ranks within each run of equal values
    sorted_x = x[order]
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or sorted_x[i] != sorted_x[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def pearson(x, y) -> float:
    """Pearson product-moment correlation. ``nan`` if either input is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y):
        return float("nan")

    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((xc**2).sum()) * float((yc**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def spearman(x, y) -> float:
    """Spearman's rank correlation coefficient."""
    return pearson(rank_average(x), rank_average(y))


def linear_fit(x, y):
    """Ordinary least-squares straight line. Returns ``(slope, intercept)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def r_squared(x, y) -> float:
    """Coefficient of determination of a straight-line fit of ``y`` on ``x``.

    For a simple linear regression this equals ``pearson(x, y) ** 2``; it is
    computed from residuals here so the definition stays explicit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")

    slope, intercept = linear_fit(x, y)
    if not math.isfinite(slope):
        return float("nan")

    residual = float(((y - (slope * x + intercept)) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0.0:
        return float("nan")
    return 1.0 - residual / total


def spearman_p_value(rho: float, n: int) -> float:
    """Two-sided p-value for ``rho`` under the t approximation.

    ``t = rho * sqrt((n - 2) / (1 - rho^2))`` on ``n - 2`` degrees of freedom.
    This approximation is asymptotic; with the ~20 epochs a training run
    provides it should be read as an order of magnitude, not a precise level.
    Returns ``nan`` when it does not apply.
    """
    if n < 3 or not math.isfinite(rho):
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0

    t = abs(rho) * math.sqrt((n - 2) / (1.0 - rho**2))
    df = n - 2
    # regularised incomplete beta via its continued fraction, through the
    # standard t-distribution survival identity
    x = df / (df + t * t)
    return _betainc(0.5 * df, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b), Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a

    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)

    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))

        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d

        c = 1.0 + numerator / c
        c = 1e-30 if abs(c) < 1e-30 else c

        delta = c * d
        f *= delta
        if abs(1.0 - delta) < 1e-12:
            break
    return front * (f - 1.0)


def running_spearman(x, y, min_points: int = 3):
    """Spearman's rho recomputed over the first ``k`` points, for ``k`` up to ``len(x)``.

    Shows whether the relationship is stable as evidence accumulates. Entries
    before ``min_points`` are ``nan`` -- a correlation over two points is always
    exactly +/-1 and means nothing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = []
    for k in range(1, len(x) + 1):
        out.append(spearman(x[:k], y[:k]) if k >= min_points else float("nan"))
    return out


def correlation_report(series: dict, pairs=None):
    """Summarise pairwise relationships between named training series.

    :param series: ``{name: values}``, e.g. ``{"epoch": [...], "loss": [...]}``.
    :param pairs: iterable of ``(a, b)`` names. Defaults to every series against
        ``"epoch"``, plus ``loss`` against every other metric.
    :returns: list of dicts with ``x``, ``y``, ``n``, ``spearman``, ``pearson``,
        ``r_squared``, ``p_value``, ``slope``.
    """
    if pairs is None:
        others = [k for k in series if k != "epoch"]
        pairs = [("epoch", k) for k in others]
        if "loss" in series:
            pairs += [("loss", k) for k in others if k != "loss"]

    rows = []
    for a, b in pairs:
        if a not in series or b not in series:
            continue
        xs, ys = series[a], series[b]
        n = min(len(xs), len(ys))
        if n < 2:
            continue
        xs, ys = xs[:n], ys[:n]

        rho = spearman(xs, ys)
        rows.append({
            "x": a,
            "y": b,
            "n": n,
            "spearman": rho,
            "pearson": pearson(xs, ys),
            "r_squared": r_squared(xs, ys),
            "p_value": spearman_p_value(rho, n),
            "slope": linear_fit(xs, ys)[0],
        })
    return rows
