"""2단계 모델 (Benter 1994 방식).

1단계: LightGBM 이진분류(1착 여부)로 '펀더멘털' 점수 -> 경주 내 softmax 로 승률화.
       배당 정보는 넣지 않는다(시장과 독립적인 의견을 만들기 위해).
2단계: 조건부 로짓  p_i ∝ exp(a·log p_fund_i + b·log p_mkt_i)
       a, b 는 1단계를 '학습에 안 쓴 기간'(내부 홀드아웃)의 예측으로 최대우도 추정.
       시장이 이미 아는 정보는 b 가, 모델만 아는 정보는 a 가 가져간다.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from features import FUND_FEATURES

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
    n_estimators=600, verbose=-1,
)


def race_softmax(score: np.ndarray, race_id: pd.Series) -> np.ndarray:
    s = pd.Series(score, index=race_id.index)
    s = s - s.groupby(race_id).transform("max")
    e = np.exp(s)
    return (e / e.groupby(race_id).transform("sum")).to_numpy()


def _clogit_nll(theta, X, race_codes, y, n_races):
    u = X @ theta
    u = u - np.maximum.reduceat(u, _starts(race_codes))[race_codes]  # 안정화
    e = np.exp(u)
    denom = np.bincount(race_codes, weights=e, minlength=n_races)
    return -(u[y == 1].sum() - np.log(denom[race_codes[y == 1]]).sum())


def _starts(codes):
    return np.r_[0, np.flatnonzero(np.diff(codes)) + 1]


def fit_clogit(X: np.ndarray, race_id: pd.Series, y: np.ndarray, l2: float = 1e-3):
    codes = pd.factorize(race_id)[0]
    order = np.argsort(codes, kind="stable")
    X, codes, y = X[order], codes[order], y[order]
    n = codes.max() + 1
    f = lambda t: _clogit_nll(t, X, codes, y, n) / n + l2 * (t @ t)
    res = minimize(f, x0=np.ones(X.shape[1]) * 0.5, method="L-BFGS-B")
    return res.x


def apply_clogit(theta, X, race_id):
    return race_softmax(X @ theta, race_id)


class TwoStageModel:
    def __init__(self, holdout_days: int = 56, use_market: bool = True, seed: int = 0):
        self.holdout_days = holdout_days
        self.use_market = use_market
        self.seed = seed

    def _fit_lgb(self, d: pd.DataFrame, n_estimators=None):
        params = dict(LGB_PARAMS, random_state=self.seed)
        if n_estimators:
            params["n_estimators"] = n_estimators
        m = lgb.LGBMClassifier(**params)
        m.fit(d[FUND_FEATURES], d["won"])
        return m

    def _fund_p(self, m, d):
        logit = m.predict(d[FUND_FEATURES], raw_score=True)
        return race_softmax(logit, d["race_id"])

    def fit(self, train: pd.DataFrame):
        train = train[train["h_starts"] >= 0]
        cut = train["date"].max() - pd.Timedelta(days=self.holdout_days)
        inner, hold = train[train["date"] <= cut], train[train["date"] > cut]

        # (1) 내부 학습 -> 홀드아웃 예측: 트리 개수 선택 + 2단계 계수 추정용
        m_inner = self._fit_lgb(inner)
        best_n, best_ll = LGB_PARAMS["n_estimators"], np.inf
        for k in (100, 200, 300, 450, 600):
            p = race_softmax(m_inner.predict(hold[FUND_FEATURES], raw_score=True, num_iteration=k),
                             hold["race_id"])
            ll = -np.log(np.clip(p[hold["won"].to_numpy() == 1], 1e-9, 1)).mean()
            if ll < best_ll:
                best_n, best_ll = k, ll
        self.n_estimators_ = best_n
        p_f = race_softmax(m_inner.predict(hold[FUND_FEATURES], raw_score=True,
                                           num_iteration=best_n), hold["race_id"])
        X = self._stage2_X(p_f, hold)
        self.theta_ = fit_clogit(X, hold["race_id"], hold["won"].to_numpy())

        # (2) 최종 1단계는 전체 학습기간으로 재학습
        self.lgb_ = self._fit_lgb(train, n_estimators=best_n)
        return self

    def _stage2_X(self, p_fund, d):
        cols = [np.log(np.clip(p_fund, 1e-6, 1))]
        if self.use_market:
            cols.append(np.log(np.clip(d["mkt_p"].to_numpy(), 1e-6, 1)))
        return np.column_stack(cols)

    def predict(self, d: pd.DataFrame) -> pd.DataFrame:
        p_f = self._fund_p(self.lgb_, d)
        p_c = apply_clogit(self.theta_, self._stage2_X(p_f, d), d["race_id"])
        return pd.DataFrame({"p_fund": p_f, "p_model": p_c}, index=d.index)

    def feature_importance(self, top=15) -> pd.Series:
        imp = pd.Series(self.lgb_.booster_.feature_importance("gain"), index=FUND_FEATURES)
        return (imp / imp.sum()).sort_values(ascending=False).head(top)


def harville_place(p: pd.Series, race_id: pd.Series, k_of: pd.Series) -> np.ndarray:
    """Harville 공식으로 '상위 k 안에 들 확률'. k = 2(7두 이하) / 3."""
    out = np.zeros(len(p))
    pos = pd.Series(np.arange(len(p)), index=p.index)
    for rid, idx in p.groupby(race_id).groups.items():
        pi = p.loc[idx].to_numpy()
        k = int(k_of.loc[idx[0]])
        n = len(pi)
        top1 = pi.copy()
        top2 = np.zeros(n)
        top3 = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                pij = pi[j] * pi[i] / (1 - pi[j])           # j 1착, i 2착
                top2[i] += pij
                if k >= 3:
                    for l in range(n):
                        if l in (i, j):
                            continue
                        top3[i] += pi[l] * pi[j] / (1 - pi[l]) * pi[i] / (1 - pi[l] - pi[j])
        res = top1 + top2 + (top3 if k >= 3 else 0)
        out[pos.loc[idx].to_numpy()] = np.clip(res, 0, 1)
    return out
