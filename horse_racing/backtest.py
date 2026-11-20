"""주간 walk-forward 백테스트.

"오늘 = 4주 전 월요일" 이라고 가정하고, 그 시점 이전 데이터만으로 학습 -> 그 주 경주 예측/베팅
-> 결과 확인 -> 한 주 전진 -> 재학습 ... 을 반복한다. 모델은 테스트 주의 결과를 절대 보지 않는다.

사용 예:
  KRA_API_KEY=... python backtest.py --fetch --today 2026-11-20 --weeks 4
  python backtest.py --csv races.csv --today 2026-11-20 --weeks 4
  python backtest.py --synthetic            # 코드 동작 확인용(가짜 데이터)
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from betting import run_strategies, summarize
from features import build_features, check_no_leakage
from model import TwoStageModel, harville_place


def logloss_win(p, d):
    return float(-np.log(np.clip(p[d["won"].to_numpy() == 1], 1e-9, 1)).mean())


def walk_forward(df: pd.DataFrame, today: pd.Timestamp, weeks: int, min_hist_days=180):
    start = today - pd.Timedelta(days=7 * weeks)
    preds, logs = [], []
    for w in range(weeks):
        cut = start + pd.Timedelta(days=7 * w)
        end = min(cut + pd.Timedelta(days=7), today)
        train = df[(df["date"] < cut) & (df["date"] >= cut - pd.Timedelta(days=730))]
        test = df[(df["date"] >= cut) & (df["date"] < end)].copy()
        if test.empty:
            continue
        assert train["date"].max() < test["date"].min(), "미래 데이터 누수"
        if (train["date"].max() - train["date"].min()).days < min_hist_days:
            raise SystemExit("학습 이력이 너무 짧음 (최소 6개월 권장)")
        m = TwoStageModel().fit(train)
        test[["p_fund", "p_model"]] = m.predict(test)
        k = test["n_runners"].map(lambda n: 2 if n <= 7 else 3)
        test["p_place"] = harville_place(test["p_model"], test["race_id"], k)
        test["week"] = f"W{w+1} {cut:%m/%d}~{end - pd.Timedelta(days=1):%m/%d}"
        preds.append(test)
        logs.append(dict(week=test["week"].iloc[0], train_rows=len(train), test_races=test["race_id"].nunique(),
                         n_trees=m.n_estimators_, theta_fund=round(m.theta_[0], 3),
                         theta_mkt=round(m.theta_[1], 3),
                         ll_market=logloss_win(test["mkt_p"].to_numpy(), test),
                         ll_fund=logloss_win(test["p_fund"].to_numpy(), test),
                         ll_model=logloss_win(test["p_model"].to_numpy(), test),
                         top_features=m.feature_importance(8).round(3).to_dict()))
        print(f"[{logs[-1]['week']}] 학습 {len(train):,}행 -> 테스트 {logs[-1]['test_races']}경주 "
              f"| logloss 시장 {logs[-1]['ll_market']:.4f} 펀더 {logs[-1]['ll_fund']:.4f} "
              f"결합 {logs[-1]['ll_model']:.4f} | θ={m.theta_.round(2)}")
    return pd.concat(preds), logs


def calibration_table(pred: pd.DataFrame, col="p_model") -> pd.DataFrame:
    bins = [0, .05, .1, .15, .2, .3, .4, .6, 1]
    g = pred.groupby(pd.cut(pred[col], bins), observed=True)
    return pd.DataFrame({"n": g.size(), "예측평균": g[col].mean(), "실제승률": g["won"].mean()})


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fetch", action="store_true", help="data.go.kr KRA API 수집 (KRA_API_KEY)")
    src.add_argument("--csv")
    src.add_argument("--synthetic", action="store_true", help="가짜 데이터로 파이프라인만 검증")
    ap.add_argument("--today", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument("--history-days", type=int, default=760)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    today = pd.Timestamp(a.today)
    if a.fetch:
        from data import fetch_kra
        raw = fetch_kra((today - timedelta(days=a.history_days)).date(), (today - timedelta(days=1)).date())
    elif a.csv:
        from data import load_csv
        raw = load_csv(a.csv)
    else:
        from synthetic import make_synthetic
        raw = make_synthetic(end=today - pd.Timedelta(days=1), days=a.history_days)
    raw = raw[raw["date"] < today]
    print(f"데이터: {raw['date'].min():%Y-%m-%d} ~ {raw['date'].max():%Y-%m-%d}, "
          f"{raw['race_id'].nunique():,}경주 / {len(raw):,}두")

    test_days = sorted(raw.loc[raw["date"] >= today - pd.Timedelta(days=7 * a.weeks), "date"].unique())
    leak = check_no_leakage(raw, test_days[len(test_days) // 2])
    print(f"누수 검사(당일 결과 조작 시 피처 변화량): {leak:.2e}")
    assert leak < 1e-9, "피처 누수 발견"

    df = build_features(raw)
    pred, logs = walk_forward(df, today, a.weeks)
    bets = run_strategies(pred)
    summ = summarize(bets)
    by_week = (bets.merge(pred[["race_id", "week"]].drop_duplicates(), on="race_id")
                   .groupby(["strategy", "week"]).apply(
                       lambda b: pd.Series({"베팅수": len(b), "적중": int(b["hit"].sum()),
                                            "손익": b["ret"].sum() - b["stake"].sum(),
                                            "ROI": b["ret"].sum() / b["stake"].sum() - 1}),
                       include_groups=False).reset_index())

    out = Path(a.out); out.mkdir(exist_ok=True)
    pred.to_csv(out / "predictions.csv", index=False)
    bets.to_csv(out / "bets.csv", index=False)
    summ.to_csv(out / "summary.csv", index=False)
    by_week.to_csv(out / "by_week.csv", index=False)
    (out / "train_log.json").write_text(json.dumps(logs, ensure_ascii=False, indent=1, default=str))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    fmt = summ.copy()
    for c in ["적중률", "ROI", "P_ROI양수", "최대낙폭"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:+.1%}" if c in ("ROI", "최대낙폭") else f"{v:.1%}")
    for c in ["총베팅", "총환수", "손익"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:,.0f}")
    print("\n=== 전략별 결과 (테스트 기간 합계) ===")
    print(fmt.to_string(index=False))
    print("\n=== 주차별 ===")
    print(by_week.assign(ROI=by_week["ROI"].map("{:+.1%}".format),
                         손익=by_week["손익"].map("{:,.0f}".format)).to_string(index=False))
    print("\n=== 결합모델 캘리브레이션 ===")
    print(calibration_table(pred).round(3).to_string())
    print("\n상위 피처(마지막 주 모델):", logs[-1]["top_features"])
    print(f"\n결과 저장: {out.resolve()}")


if __name__ == "__main__":
    main()
