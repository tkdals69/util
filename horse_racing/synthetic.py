"""가짜 경주 데이터 생성기 — 파이프라인/누수검사 동작 확인 전용.
여기서 나온 수익률은 실제 경마 성과와 아무 관계가 없다."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import normalize
from model import harville_place


def make_synthetic(end, days=760, seed=7, takeout=0.20):
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end)
    n_h, n_j, n_t = 2500, 70, 55
    ability = rng.normal(0, 1, n_h)
    dist_pref = rng.choice([1000, 1200, 1400, 1600, 1800, 2000], n_h)
    jk = rng.normal(0, 0.35, n_j)
    tr = rng.normal(0, 0.3, n_t)
    h_tr = rng.integers(0, n_t, n_h)
    rating = np.clip(60 + 15 * ability + rng.normal(0, 5, n_h), 10, 120)
    wt = rng.normal(470, 25, n_h)
    rows = []
    for d in pd.date_range(end - pd.Timedelta(days=days), end):
        if d.dayofweek not in (4, 5, 6):
            continue
        ability += rng.normal(0, 0.03, n_h)                  # 컨디션 드리프트
        meets = {4: [2, 3], 5: [1, 2], 6: [1, 3]}[d.dayofweek]
        day_pool = rng.permutation(n_h)              # 하루에 한 말은 한 번만 출주
        for mi, meet in enumerate(meets):
            pool = day_pool[mi * 160:(mi + 1) * 160]
            pos = 0
            for rno in range(1, 11):
                n = int(rng.integers(7, 15))
                hs = np.sort(pool[pos:pos + n]); pos += n
                hs = hs[np.argsort(rating[hs])]                # 등급 비슷한 말끼리
                dist = rng.choice([1000, 1200, 1400, 1600, 1800, 2000])
                js = rng.choice(n_j, n, replace=False)
                gates = rng.permutation(n) + 1
                carried = 52 + (rating[hs] - rating[hs].mean()) / 8 + rng.normal(0, 1, n)
                perf = (ability[hs] + jk[js] + tr[h_tr[hs]] - 0.02 * gates
                        - 0.0006 * np.abs(dist - dist_pref[hs]) - 0.08 * (carried - 54))
                # 시장은 능력/기수는 잘 알지만 노이즈가 있고, 인기마 쏠림(favorite-longshot bias)
                view = 1.15 * (ability[hs] + jk[js] + 0.5 * tr[h_tr[hs]]) + rng.normal(0, 0.45, n)
                q = np.exp(view - view.max()); q /= q.sum()
                q = q ** 0.9; q /= q.sum()
                win_odds = np.maximum(1.0, np.round((1 - takeout) / q, 1))
                kq = 2 if n <= 7 else 3
                qp = harville_place(pd.Series(q), pd.Series(np.zeros(n)), pd.Series(np.full(n, kq)))
                place_odds = np.maximum(1.0, np.round((1 - takeout) / qp, 1))
                outcome = perf + rng.gumbel(0, 0.9, n)
                order = np.empty(n, int); order[np.argsort(-outcome)] = np.arange(1, n + 1)
                base = dist / 16.5
                t = base + (outcome.max() - outcome) * 0.6 + rng.normal(0, 0.05, n)
                for i in range(n):
                    h = hs[i]
                    wt[h] += rng.normal(0, 4)
                    rows.append(dict(rcDate=d.strftime("%Y%m%d"), meet=meet, rcNo=rno, chulNo=gates[i],
                                     hrNo=f"H{h:05d}", hrName=f"말{h}", jkNo=f"J{js[i]:03d}",
                                     trNo=f"T{h_tr[h]:03d}", ord=order[i], rcDist=dist,
                                     wgBudam=round(carried[i], 1),
                                     wgHr=f"{wt[h]:.0f}({rng.integers(-8, 9):+d})",
                                     age=int(rng.integers(2, 7)), sex=rng.choice(["수", "암", "거"]),
                                     rating=round(rating[h]), track=f"건조 ({rng.integers(2, 15)}%)",
                                     weather="맑음", winOdds=win_odds[i], plcOdds=place_odds[i],
                                     rcTime=f"{t[i]:.1f}"))
                rating[hs] += (np.median(order) - order) * 0.4    # 성적 반영 레이팅 조정
    return normalize(pd.DataFrame(rows))
