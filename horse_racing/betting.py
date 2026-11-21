"""베팅 전략. 배당은 KRA 배당률(원금 포함 배수, 예: 3.4 -> 1,000원이 3,400원).

주의: 백테스트는 '확정 배당'으로 정산한다. 실전에서는 마감 직전 배당으로 사야 하고
대량 베팅은 자기 돈으로 배당을 깎는다(파리뮤추얼). 그래서 소액·고정 가정이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNIT = 10_000          # 고정 베팅 단위(원)
MIN_BET = 100


def _round(x):
    return np.floor(x / MIN_BET) * MIN_BET


def flat_favorite(race: pd.DataFrame, **_):
    i = race["win_odds"].idxmin()
    return [(i, "win", UNIT)]


def flat_model_top(race: pd.DataFrame, **_):
    i = race["p_model"].idxmax()
    return [(i, "win", UNIT)]


def value_win(race: pd.DataFrame, edge=0.10, min_p=0.04, max_odds=40.0, **_):
    ev = race["p_model"] * race["win_odds"]
    pick = race[(ev > 1 + edge) & (race["p_model"] >= min_p) & (race["win_odds"] <= max_odds)]
    return [(i, "win", UNIT) for i in pick.index]


def value_place(race: pd.DataFrame, edge=0.08, **_):
    if race["place_odds"].isna().all():
        return []
    ev = race["p_place"] * race["place_odds"]
    pick = race[(ev > 1 + edge) & (race["p_place"] >= 0.25)]
    return [(i, "place", UNIT) for i in pick.index]


def kelly_win(race: pd.DataFrame, bankroll: float, frac=0.25, cap=0.02, race_cap=0.05,
              min_edge=0.05, max_odds=40.0, **_):
    o, p = race["win_odds"], race["p_model"]
    f = (p * o - 1) / (o - 1)
    f = f.where((p * o > 1 + min_edge) & (o <= max_odds) & (o > 1), 0).clip(lower=0) * frac
    f = f.clip(upper=cap)
    if f.sum() > race_cap:
        f *= race_cap / f.sum()
    return [(i, "win", _round(bankroll * fi)) for i, fi in f.items() if _round(bankroll * fi) > 0]


def favorite_place(race: pd.DataFrame, **_):
    """비교 기준: 시장 1번 인기마 연승."""
    i = race["win_odds"].idxmin()
    return [(i, "place", UNIT)]


def hc_place(race: pd.DataFrame, **_):
    """고적중 연승: 경주에서 p_place 최고 말이 임계값 이상일 때만.
    임계값(hc_thr)은 walk_forward 에서 학습기간 홀드아웃으로만 정한다."""
    i = race["p_place"].idxmax()
    if race.loc[i, "p_place"] >= race["hc_thr"].iloc[0] and race.loc[i, "place_odds"] > 1.0:
        return [(i, "place", UNIT)]
    return []


def hc_dutch(race: pd.DataFrame, target=0.75, max_horses=4, **_):
    """고적중 단승 더칭: 확률 높은 순으로 누적 p_model >= target 될 때까지 여러 마리를 사고,
    어느 말이 와도 환수액이 같도록 stake ∝ 1/odds 로 나눈다."""
    r = race.sort_values("p_model", ascending=False)
    cum = r["p_model"].cumsum()
    k = int((cum < target).sum()) + 1
    if k > max_horses:
        return []
    pick = r.iloc[:k]
    inv = 1 / pick["win_odds"]
    if inv.sum() >= 1.0:
        return []                     # 맞혀도 원금 이하(배당 합성 1배 이하)면 안 산다
    stakes = UNIT * len(pick) * inv / inv.sum()
    return [(i, "win", _round(st)) for i, st in stakes.items()]


STRATEGIES = {
    "S0_인기마_고정": flat_favorite,
    "S1_모델1위_고정": flat_model_top,
    "S2_밸류단승_고정": value_win,
    "S3_밸류연승_고정": value_place,
    "S4_켈리단승_1/4": kelly_win,
    "S5_인기마_연승": favorite_place,
    "S6_고적중_연승": hc_place,
    "S7_고적중_단승더칭": hc_dutch,
}


def run_strategies(pred: pd.DataFrame, start_bankroll=1_000_000) -> pd.DataFrame:
    """경주 시간 순으로 진행하며 모든 전략의 베팅 기록을 반환."""
    rows = []
    bank = {k: float(start_bankroll) for k in STRATEGIES}
    for rid, race in pred.sort_values(["date", "race_id"]).groupby("race_id", sort=False):
        for name, fn in STRATEGIES.items():
            for i, kind, stake in fn(race, bankroll=bank[name]):
                if stake <= 0 or stake > bank[name]:
                    continue
                r = race.loc[i]
                hit = bool(r["won"]) if kind == "win" else bool(r["placed"])
                odds = r["win_odds"] if kind == "win" else r["place_odds"]
                ret = stake * odds if hit else 0.0
                bank[name] += ret - stake
                rows.append(dict(strategy=name, date=r["date"], race_id=rid, horse=r.get("horse"),
                                 gate=r["gate"], kind=kind, p=r["p_model"] if kind == "win" else r["p_place"],
                                 odds=odds, stake=stake, hit=hit, ret=ret, bankroll=bank[name]))
    return pd.DataFrame(rows)


def summarize(bets: pd.DataFrame, n_boot=2000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for name, b in bets.groupby("strategy", sort=False):
        per_race = b.groupby("race_id")[["stake", "ret"]].sum()
        roi = per_race["ret"].sum() / per_race["stake"].sum() - 1
        boots = []
        arr = per_race.to_numpy()
        for _ in range(n_boot):
            s = arr[rng.integers(0, len(arr), len(arr))]
            boots.append(s[:, 1].sum() / s[:, 0].sum() - 1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        curve = b["bankroll"].to_numpy()
        dd = (curve / np.maximum.accumulate(curve) - 1).min() if len(curve) else 0
        race_hit = b.groupby("race_id")["hit"].any()
        out.append(dict(전략=name, 베팅경주=len(race_hit), 경주적중률=race_hit.mean(), 베팅수=len(b), 총베팅=b["stake"].sum(),
                        총환수=b["ret"].sum(), 손익=b["ret"].sum() - b["stake"].sum(), ROI=roi,
                        ROI_95CI=f"[{lo:+.1%}, {hi:+.1%}]", P_ROI양수=float(np.mean(np.array(boots) > 0)),
                        평균배당=b["odds"].mean(), 최대낙폭=dd))
    return pd.DataFrame(out)
