"""데이터 수집/정규화.

공식 소스: 공공데이터포털(data.go.kr) 한국마사회 '경주기록정보' OpenAPI.
  - serviceKey 필요 (data.go.kr 에서 무료 발급, 환경변수 KRA_API_KEY)
  - meet: 1=서울, 2=제주, 3=부산경남
API 응답 필드명은 버전마다 조금씩 달라서 COLUMN_ALIASES 로 흡수한다.
수집이 안 되는 환경이면 같은 스키마의 CSV 를 --csv 로 넣으면 된다.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

KRA_ENDPOINT = "https://apis.data.go.kr/B551015/API214_1/RaceDetailResult_1"
CACHE_DIR = Path(os.environ.get("KRA_CACHE", "./cache"))

# 정규화 스키마 -> 가능한 원본 필드명
COLUMN_ALIASES = {
    "date": ["rcDate", "rc_date", "date"],
    "meet": ["meet"],
    "race_no": ["rcNo", "rc_no", "race_no"],
    "gate": ["chulNo", "chul_no", "gate"],
    "horse_id": ["hrNo", "hr_no", "horse_id"],
    "horse": ["hrName", "hr_name", "horse"],
    "jockey_id": ["jkNo", "jk_no", "jockey_id"],
    "trainer_id": ["trNo", "tr_no", "trainer_id"],
    "finish_pos": ["ord", "rcOrd", "finish_pos"],
    "distance": ["rcDist", "rc_dist", "distance"],
    "carried_wt": ["wgBudam", "wg_budam", "carried_wt"],
    "horse_wt_raw": ["wgHr", "wg_hr", "horse_wt_raw"],
    "age": ["age"],
    "sex": ["sex"],
    "rating": ["rating"],
    "track": ["track", "trackCond"],
    "weather": ["weather"],
    "race_grade": ["rank", "rcGrade", "race_grade"],
    "win_odds": ["winOdds", "win_odds"],
    "place_odds": ["plcOdds", "plc_odds", "place_odds"],
    "race_time": ["rcTime", "rc_time", "race_time"],
}

REQUIRED = ["date", "meet", "race_no", "horse_id", "finish_pos", "win_odds"]


def _fetch_day(day: date, meet: int, key: str, session) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{meet}_{day:%Y%m%d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    rows, page = [], 1
    while True:
        params = {"serviceKey": key, "pageNo": page, "numOfRows": 1000,
                  "meet": meet, "rc_date": f"{day:%Y%m%d}", "_type": "json"}
        for attempt in range(4):
            try:
                r = session.get(KRA_ENDPOINT, params=params, timeout=30)
                r.raise_for_status()
                body = r.json()["response"]["body"]
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        items = (body.get("items") or {}) if isinstance(body.get("items"), dict) else {}
        item = items.get("item", [])
        if isinstance(item, dict):
            item = [item]
        rows.extend(item)
        if len(rows) >= int(body.get("totalCount", 0)) or not item:
            break
        page += 1
    cache.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def fetch_kra(start: date, end: date, meets=(1, 2, 3), key: str | None = None) -> pd.DataFrame:
    """start~end 기간 금/토/일(+임시 경마일 대비 전 요일) 결과 수집."""
    import requests

    key = key or os.environ["KRA_API_KEY"]
    s = requests.Session()
    out, d = [], start
    while d <= end:
        for m in meets:
            out.extend(_fetch_day(d, m, key, s))
        d += timedelta(days=1)
    return normalize(pd.DataFrame(out))


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    for col, aliases in COLUMN_ALIASES.items():
        src = next((a for a in aliases if a in raw.columns), None)
        df[col] = raw[src] if src else pd.NA
    missing = [c for c in REQUIRED if df[c].isna().all()]
    if missing:
        raise ValueError(f"필수 컬럼 없음: {missing}. 원본 컬럼: {list(raw.columns)}")

    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    for c in ["meet", "race_no", "gate", "finish_pos", "distance", "carried_wt",
              "age", "rating", "win_odds", "place_odds"]:
        df[c] = _to_num(df[c])
    # 마체중 "470(+3)" -> 470, +3
    hw = df["horse_wt_raw"].astype(str)
    df["horse_wt"] = _to_num(hw.str.extract(r"^(\d+)")[0])
    df["horse_wt_chg"] = _to_num(hw.str.extract(r"\(([-+]?\d+)\)")[0])
    df = df.drop(columns="horse_wt_raw")
    for c in ["horse_id", "jockey_id", "trainer_id", "sex", "track", "weather", "race_grade"]:
        df[c] = df[c].astype(str)

    df = df.dropna(subset=["date", "meet", "race_no", "horse_id"])
    # 실격/취소(순위 0 또는 결측)는 출전 안 한 것으로 본다
    df = df[df["finish_pos"].fillna(0) > 0].copy()
    df["race_id"] = (df["date"].dt.strftime("%Y%m%d") + "_" + df["meet"].astype(int).astype(str)
                     + "_" + df["race_no"].astype(int).astype(str).str.zfill(2))
    df["n_runners"] = df.groupby("race_id")["horse_id"].transform("size")
    df["won"] = (df["finish_pos"] == 1).astype(int)
    df["placed"] = (df["finish_pos"] <= df["n_runners"].map(lambda n: 2 if n <= 7 else 3)).astype(int)
    df = df[df["n_runners"] >= 4]
    dup = df.duplicated(["horse_id", "date"], keep=False)
    if dup.any():  # 같은 말이 같은 날 2회 -> 말 이력 피처가 당일 결과를 볼 수 있으므로 제거
        print(f"[warn] 같은 날 중복 출주 {dup.sum()}행 제거")
        df = df[~dup]
    return df.sort_values(["date", "race_id", "gate"]).reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    return normalize(pd.read_csv(path, dtype=str))
