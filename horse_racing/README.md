# 경마 walk-forward 백테스트 (KRA)

"오늘이 N주 전"이라고 가정하고, 그날 이전 데이터만으로 학습 → 그 주 경주에 베팅 → 결과 확인 →
1주 전진·재학습을 반복하는 백테스트. 기본은 최근 4주(한 달).

## 실행
```bash
pip install -r requirements.txt
# 실데이터: data.go.kr 에서 '한국마사회 경주기록정보' 활용신청 후 키 발급
KRA_API_KEY=발급키 python backtest.py --fetch --today 2026-11-20 --weeks 4
# 또는 직접 받은 CSV (KRA 원본 필드명 그대로 OK: rcDate, meet, rcNo, chulNo, hrNo, ord, winOdds, plcOdds ...)
python backtest.py --csv races.csv --today 2026-11-20 --weeks 4
# 코드 동작 확인용 가짜 데이터 (수익률은 의미 없음)
python backtest.py --synthetic
```
결과: `results/summary.csv, by_week.csv, bets.csv, predictions.csv, train_log.json`

## 구조
| 파일 | 내용 |
|---|---|
| `data.py` | KRA OpenAPI 수집(캐시) + 필드 정규화, 마체중 `470(+3)` 파싱, 실격 제거 |
| `features.py` | 시점 기준 피처 ~70개: 말 이력(최근 착순, 착차/km, EWM, 동일거리), 기수·조교사·콤비 베이지안 승률, 시장 대비 초과성과(edge), 경주 내 z-score/순위 |
| `model.py` | 1단계 LightGBM(배당 미사용) → 경주 내 softmax, 2단계 조건부 로짓으로 시장 확률과 결합(Benter 방식), Harville 연승 확률 |
| `betting.py` | 인기마/모델1위/밸류 단승/밸류 연승/1/4 켈리 + 부트스트랩 ROI 신뢰구간, 최대낙폭 |
| `backtest.py` | 주간 walk-forward, 누수 검사, 로그손실(시장 vs 모델), 캘리브레이션 |

## 누수 방지
- 학습셋 최대 날짜 < 테스트 최소 날짜 assert
- 같은 날 앞 경주 결과도 사용 안 함(기수/조교사는 날짜 단위 누적 후 당일 제외)
- `check_no_leakage`: 테스트일 결과를 조작해도 그날 피처가 0 오차로 동일한지 매번 검사

## 해석 시 주의
- **확정 배당으로 정산**한다. 실전에서는 마감 전 배당으로 사므로 결과가 더 나쁠 수 있다.
- 한 달 ≈ 200~250경주. 부트스트랩 CI가 넓으면 운인지 실력인지 구분 불가.
- 2단계 θ_fund 가 0 근처면 "모델이 시장보다 아는 게 없다"는 뜻 → 밸류 베팅이 거의 안 나와야 정상.
- 단승/연승 환급률 약 80% → 아무 정보 없이 사면 장기 ROI ≈ -20%.

## 고적중(적중률 70~80%) 전략
- `S5_인기마_연승`: 비교 기준. 시장 1번 인기마 연승.
- `S6_고적중_연승`: 경주별 Harville 연승확률 최고 말, 임계값 이상일 때만. 임계값은 **매주 학습기간 홀드아웃**에서
  목표 적중률(`--target-hit`, 기본 0.75)을 맞추는 값으로 고른다(테스트 결과 미사용).
- `S7_고적중_단승더칭`: 누적 승률 75%가 될 때까지 최대 4두를 단승으로 나눠 사서 어느 말이 와도 같은 금액 환수.
- `경주적중률` = 한 경주에서 하나라도 맞힌 비율. 적중률과 수익률은 별개다. 배당 1.1배면 91% 적중해야 본전.
