"""Shared scoring: keyed joins, fractional ties, constant signals held in cash."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def align_predictions(panel, predictions):
    """Economic keys remain valid when a panel's row index changes."""
    keys = ["date", "permno"]
    p = predictions[keys + ["prediction"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    panel_keys = panel[keys].copy()
    panel_keys["date"] = pd.to_datetime(panel_keys["date"])
    for name, frame in [("Prediction", p), ("Panel", panel_keys)]:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name} keys must be nonmissing and unique (date, permno)")
    idx = pd.MultiIndex.from_frame(panel_keys)
    values = p.set_index(keys)["prediction"].reindex(idx).to_numpy(dtype=float)
    return pd.Series(values, index=panel.index).replace([np.inf, -np.inf], np.nan)


def rank_correlation(df, prediction, target, by=None, min_obs=10):
    by = ["date"] if by is None else list(by)
    d = df[by + [prediction, target]].replace([np.inf, -np.inf], np.nan).dropna()
    g = d.groupby(by, observed=True)
    ranks = g[[prediction, target]].rank()
    for c in by:
        ranks[c] = d[c]
    rg = ranks.groupby(by, observed=True)
    x = ranks[prediction] - rg[prediction].transform("mean")
    y = ranks[target] - rg[target].transform("mean")
    sums = d[by].assign(xx=x*x, yy=y*y, xy=x*y, n=1).groupby(by, observed=True).sum()
    den = np.sqrt(sums.xx*sums.yy)
    rc = (sums.xy/den.where(den > 0)).where(sums.xx > 0, 0.0)
    return rc.where((sums.n >= min_obs) & (sums.yy > 0))


def bin_returns(df, prediction, target, bins=10, weight=None, min_per_bin=1):
    """Ties share their rank interval across bins, independent of row order.

    A tied block occupies [a,b]; each member gets the fraction of that interval
    overlapping a bin. Cap weights apply after allocation. Counts are fractional
    memberships. An eligible day with a constant signal holds cash; missing
    signals are excluded. Constant targets have undefined IC, but valid returns.
    """
    if bins < 2 or min_per_bin < 1:
        raise ValueError("Require two bins and at least one security per bin")
    cols = list(dict.fromkeys(["date", prediction, target] + ([weight] if weight else [])))
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if weight:
        d = d[d[weight] > 0]
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(df.date).unique()), name="date")
    if d.empty:
        empty = pd.DataFrame(np.nan, index=dates, columns=range(1, bins+1))
        return empty, empty.copy()
    g = d.groupby("date")
    left = g[prediction].rank(method="min").to_numpy()-1
    right = g[prediction].rank(method="max").to_numpy()
    n = g[prediction].transform("size").to_numpy()
    codes, observed_dates = pd.factorize(d.date, sort=True)
    w = d[weight].to_numpy(dtype=float) if weight else np.ones(len(d))
    ret = d[target].to_numpy(dtype=float)
    profiles, counts = {}, {}
    for b in range(1, bins+1):
        frac = np.maximum(0, np.minimum(right, b*n/bins)-np.maximum(left, (b-1)*n/bins))/(right-left)
        mass = np.bincount(codes, weights=frac*w)
        total = np.bincount(codes, weights=frac*w*ret)
        profiles[b] = np.divide(total, mass, out=np.full(len(mass), np.nan), where=mass > 0)
        counts[b] = np.bincount(codes, weights=frac)
    profiles = pd.DataFrame(profiles, index=pd.DatetimeIndex(observed_dates, name="date"))
    counts = pd.DataFrame(counts, index=profiles.index)
    eligible = counts.min(axis=1) >= min_per_bin-1e-9
    constant = g[prediction].nunique().reindex(profiles.index) == 1
    profiles.loc[constant & eligible, :] = 0.0
    profiles.loc[~eligible, :] = np.nan
    return profiles.reindex(dates), counts.reindex(dates)


def hac_mean(series, lags=5):
    """Bartlett/Newey-West SE; two-sided normal inference with N/(N-1) correction.

    Lags count observed trading dates after missing values are removed.
    Disjoint pilot periods must be scored separately.
    """
    if lags < 0:
        raise ValueError("HAC lags must be nonnegative")
    x = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    out = {"mean": x.mean(), "se": np.nan, "t": np.nan, "p": np.nan,
           "ci_low": np.nan, "ci_high": np.nan, "n": len(x)}
    if len(x) < 2:
        return out
    if np.ptp(x.to_numpy()) == 0:
        out.update(se=0.0, ci_low=x.iloc[0], ci_high=x.iloc[0])
        if x.iloc[0] == 0:
            out.update(t=0.0, p=1.0)
        return out
    result = sm.OLS(x.to_numpy(), np.ones((len(x), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": min(lags, len(x)-1), "use_correction": True}, use_t=False)
    mean, se = float(result.params[0]), float(result.bse[0])
    out.update(se=se, t=float(result.tvalues[0]), p=float(result.pvalues[0]),
               ci_low=mean-norm.ppf(.975)*se, ci_high=mean+norm.ppf(.975)*se)
    return out


def sharpe(series):
    x = pd.Series(series).dropna()
    return x.mean()/x.std()*np.sqrt(252) if len(x) > 1 and x.std() > 0 else np.nan
