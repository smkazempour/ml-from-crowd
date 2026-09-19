"""Reusable, immutable target-rank caches for repeated daily NN validation IC.

Build each cache exclusively from its own temporal block. Changing prediction
values never alters the cache. Missing predictions trigger target re-ranking on
the surviving observations, matching prediction_metrics.rank_correlation.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _DateRanks:
    indices: np.ndarray
    target: np.ndarray
    centered_rank: np.ndarray
    squared_norm: float


class DailyICScorer:
    """Cache target ranks for mean daily Spearman correlation.

    ``score(prediction)`` returns the mean across eligible dates; ``daily``
    returns the individual date values. A constant prediction gets zero IC;
    a constant target or fewer than ``min_obs`` matched rows gets undefined IC.
    Target arrays are copied and made read-only so caller mutations cannot
    silently change the validation criterion midway through a training run.
    """

    def __init__(self, target, codes, min_obs=10):
        target, codes = np.asarray(target, dtype=float), np.asarray(codes)
        if target.ndim != 1 or codes.shape != target.shape:
            raise ValueError("Target and date codes must be equal-length vectors")
        if min_obs < 1:
            raise ValueError("min_obs must be positive")
        self.n_rows, self.min_obs = len(target), int(min_obs)
        valid_codes = pd.Series(codes).replace([np.inf, -np.inf], np.nan).notna().to_numpy()
        eligible = np.flatnonzero(np.isfinite(target) & valid_codes)
        date_codes, dates = pd.factorize(codes[eligible], sort=True)
        order = np.argsort(date_codes, kind="stable")
        eligible = eligible[order]
        sorted_codes = date_codes[order]
        bounds = np.r_[0, np.flatnonzero(np.diff(sorted_codes))+1, len(order)]
        groups = []
        if len(order):
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                rows = eligible[lo:hi]
                values = target[rows]
                ranks = rankdata(values, method="average")-(len(rows)+1)/2
                groups.append(_DateRanks(_readonly(rows, np.int64), _readonly(values),
                                         _readonly(ranks), float(ranks@ranks)))
        self._groups = tuple(groups)
        self._dates = pd.Index(dates, name="date")

    def _values(self, prediction):
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != (self.n_rows,):
            raise ValueError("Prediction length does not match the cached target block")
        finite_everywhere = np.isfinite(prediction).all()
        values = np.full(len(self._groups), np.nan)
        present = np.ones(len(self._groups), dtype=bool)
        for j, group in enumerate(self._groups):
            x = prediction[group.indices]
            y_rank, y_squared = group.centered_rank, group.squared_norm
            if not finite_everywhere:
                finite = np.isfinite(x)
                present[j] = bool(finite.any())
                if not finite.all():
                    x = x[finite]
                    if len(x) < self.min_obs:
                        continue
                    # Retained targets must be re-ranked when a prediction is
                    # missing; using their original ranks would change Spearman.
                    y_rank = rankdata(group.target[finite], method="average")-(len(x)+1)/2
                    y_squared = float(y_rank@y_rank)
            if len(x) < self.min_obs or y_squared == 0:
                continue
            x_rank = rankdata(x, method="average")-(len(x)+1)/2
            x_squared = float(x_rank@x_rank)
            values[j] = (float(x_rank@y_rank)/np.sqrt(x_squared*y_squared)
                         if x_squared > 0 else 0.)
        return values, present

    def score(self, prediction):
        values, _ = self._values(prediction)
        valid = np.isfinite(values)
        return float(values[valid].mean()) if valid.any() else float("nan")

    def daily(self, prediction):
        values, present = self._values(prediction)
        return pd.Series(values[present], index=self._dates[present], name="rank_ic")

    __call__ = score
