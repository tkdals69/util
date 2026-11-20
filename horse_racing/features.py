"""시점 기준(point-in-time) 피처.

모든 과거 통계는 '해당 경주일 이전' 데이터만 사용한다. 같은 날 앞 경주 결과도 쓰지 않는다
(실전에서 아침에 베팅 계획을 세운다고 가정). check_no_leakage() 로 검증 가능.
배당(win_odds)은 확정배당이므로 2단계(시장 결합)에서만 쓰고 펀더멘털 모델에는 넣지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_WIN = 0.10   # 베이지안 수축용 사전 승률
PRIOR_N = 20.0


def _parse_time(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    mmss = s.str.extract(r"^(\d+):(\d+\.?\d*)$")
    t = pd.to_numeric(mmss[0], errors="coerce") * 60 + pd.to_numeric(mmss[1], errors="coerce")
    return t.fillna(pd.to_numeric(s, errors="coerce"))


def market_prob(df: pd.DataFrame) -> pd.Series:
    inv = 1.0 / df["win_odds"].where(df["win_odds"] > 0)
    inv = inv.fillna(inv.groupby(df["race_id"]).transform("min"))  # 배당 결측 -> 최저값
    return inv / inv.groupby(df["race_id"]).transform("sum")


def _entity_rates(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """기수/조교사 등: 날짜 단위로 집계 후 '전일까지' 누적. 같은 날 결과 미사용."""
    g = (df.groupby([key, "date"])
           .agg(n=("won", "size"), w=("won", "sum"), p=("placed", "sum"),
                exp=("mkt_p", "sum"))
           .reset_index().sort_values([key, "date"]))
    for c in ["n", "w", "p", "exp"]:
        g[f"cum_{c}"] = g.groupby(key)[c].cumsum() - g[c]
    out = pd.DataFrame({
        key: g[key], "date": g["date"],
        f"{prefix}_n": g["cum_n"],
        f"{prefix}_win": (g["cum_w"] + PRIOR_WIN * PRIOR_N) / (g["cum_n"] + PRIOR_N),
        f"{prefix}_place": (g["cum_p"] + 0.3 * PRIOR_N) / (g["cum_n"] + PRIOR_N),
        # 시장 기대 대비 초과 승수 (시장이 과소평가하는 기수인가)
        f"{prefix}_edge": (g["cum_w"] - g["cum_exp"]) / (g["cum_n"] + PRIOR_N),
    })
    return df[[key, "date"]].merge(out, on=[key, "date"], how="left")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["date", "race_id", "gate"]).reset_index(drop=True).copy()
    df["mkt_p"] = market_prob(df)
    n = df["n_runners"]
    df["fin_pct"] = (df["finish_pos"] - 1) / (n - 1)          # 0=1착, 1=꼴찌
    rt = _parse_time(df["race_time"]) if "race_time" in df else pd.Series(np.nan, index=df.index)
    win_t = rt.groupby(df["race_id"]).transform("min")
    df["behind_per_km"] = (rt - win_t) / (df["distance"] / 1000)

    # ---- 말 단위: 말 이력은 하루 1경주라 shift(1)이면 충분
    df = df.sort_values(["horse_id", "date"]).copy()
    h = df.groupby("horse_id", sort=False)
    df["h_starts"] = h.cumcount()
    for col, name in [("won", "h_wins"), ("placed", "h_places"), ("mkt_p", "h_exp")]:
        df[name] = h[col].cumsum() - df[col]
    df["h_win_rate"] = (df["h_wins"] + PRIOR_WIN * 5) / (df["h_starts"] + 5)
    df["h_place_rate"] = (df["h_places"] + 0.3 * 5) / (df["h_starts"] + 5)
    df["h_edge"] = (df["h_wins"] - df["h_exp"]) / (df["h_starts"] + 5)
    for k in (1, 2, 3):
        df[f"h_fin_last{k}"] = h["fin_pct"].shift(k)
        df[f"h_behind_last{k}"] = h["behind_per_km"].shift(k)
    df["h_fin_ewm"] = h["fin_pct"].transform(lambda s: s.shift(1).ewm(halflife=3).mean())
    df["h_behind_ewm"] = h["behind_per_km"].transform(lambda s: s.shift(1).ewm(halflife=3).mean())
    df["h_mktp_ewm"] = h["mkt_p"].transform(lambda s: s.shift(1).ewm(halflife=3).mean())
    df["h_days_off"] = (df["date"] - h["date"].shift(1)).dt.days
    df["h_dist_chg"] = df["distance"] - h["distance"].shift(1)
    df["h_rating_chg"] = df["rating"] - h["rating"].shift(1)
    df["h_carried_chg"] = df["carried_wt"] - h["carried_wt"].shift(1)
    df["h_runners_last"] = h["n_runners"].shift(1)
    # 같은 거리 과거 성적
    hd = df.groupby(["horse_id", "distance"], sort=False)
    df["h_same_dist_n"] = hd.cumcount()
    df["h_same_dist_fin"] = hd["fin_pct"].transform(lambda s: s.shift(1).expanding().mean())

    df = df.sort_values(["date", "race_id", "gate"]).reset_index(drop=True)
    for key, pre in [("jockey_id", "jk"), ("trainer_id", "tr")]:
        r = _entity_rates(df, key, pre)
        for c in r.columns.difference([key, "date"]):
            df[c] = r[c].values
    # 기수-조교사 콤비
    df["combo_id"] = df["jockey_id"] + "|" + df["trainer_id"]
    r = _entity_rates(df, "combo_id", "combo")
    df["combo_win"], df["combo_n"] = r["combo_win"].values, r["combo_n"].values

    # ---- 기본 정적 피처
    df["gate_rel"] = df["gate"] / n
    df["sex_code"] = df["sex"].astype("category").cat.codes
    df["track_moist"] = pd.to_numeric(df["track"].str.extract(r"(\d+)")[0], errors="coerce")

    # ---- 경주 내 상대화: 절대값보다 '이 경주 상대들 대비' 가 중요
    rel_cols = ["rating", "h_win_rate", "h_fin_ewm", "h_behind_ewm", "h_mktp_ewm", "jk_win",
                "tr_win", "carried_wt", "horse_wt", "h_edge", "jk_edge", "age"]
    g = df.groupby("race_id")
    for c in rel_cols:
        mu, sd = g[c].transform("mean"), g[c].transform("std")
        df[f"{c}_z"] = (df[c] - mu) / sd.replace(0, np.nan)
        df[f"{c}_rk"] = g[c].rank(pct=True)
    return df


FUND_FEATURES = [
    "gate", "gate_rel", "n_runners", "distance", "carried_wt", "horse_wt", "horse_wt_chg",
    "age", "sex_code", "rating", "track_moist", "meet",
    "h_starts", "h_win_rate", "h_place_rate", "h_edge", "h_fin_last1", "h_fin_last2",
    "h_fin_last3", "h_behind_last1", "h_behind_last2", "h_behind_last3", "h_fin_ewm",
    "h_behind_ewm", "h_mktp_ewm", "h_days_off", "h_dist_chg", "h_rating_chg",
    "h_carried_chg", "h_runners_last", "h_same_dist_n", "h_same_dist_fin",
    "jk_n", "jk_win", "jk_place", "jk_edge", "tr_n", "tr_win", "tr_place", "tr_edge",
    "combo_win", "combo_n",
] + [f"{c}_{s}" for c in ["rating", "h_win_rate", "h_fin_ewm", "h_behind_ewm", "h_mktp_ewm",
                          "jk_win", "tr_win", "carried_wt", "horse_wt", "h_edge", "jk_edge", "age"]
     for s in ("z", "rk")]


def check_no_leakage(df_raw: pd.DataFrame, cutoff, sample: int = 300, seed: int = 0) -> float:
    """cutoff 당일 경주의 피처를, (a) 전체 데이터로 만든 것과 (b) cutoff 당일까지만 남기고
    결과 컬럼을 지운 데이터로 만든 것이 같은지 확인. 최대 절대 오차를 반환(0이어야 함)."""
    cutoff = pd.Timestamp(cutoff)
    full = build_features(df_raw)
    trunc_raw = df_raw[df_raw["date"] <= cutoff].copy()
    today = trunc_raw["date"] == cutoff
    # 당일 결과를 엉터리 값으로 바꿔도 당일 피처가 변하면 안 된다
    trunc_raw.loc[today, ["finish_pos", "won", "placed"]] = 1
    if "race_time" in trunc_raw:
        trunc_raw.loc[today, "race_time"] = "99"
    trunc = build_features(trunc_raw)
    cols = FUND_FEATURES
    key = ["race_id", "horse_id"]
    a = full[full["date"] == cutoff].set_index(key)[cols]
    b = trunc[trunc["date"] == cutoff].set_index(key)[cols]
    a = a.loc[b.index].sample(min(sample, len(b)), random_state=seed)
    diff = (a - b.loc[a.index]).abs().max().max()
    return float(np.nan_to_num(diff))
