import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 로드 및 전처리
url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
df = pd.read_csv(url, encoding="utf-8")

df["날짜"] = pd.to_datetime(df["날짜"])
df["연도"] = df["날짜"].dt.year

# 2025년 이하 및 연간 300일 이상 관측 연도 필터링
df = df[df["연도"] <= 2025]
valid_counts = df.dropna(subset=["평균기온"]).groupby("연도").size()
valid_years = valid_counts[valid_counts >= 300].index

df_valid = df[df["연도"].isin(valid_years)]
annual_temp = df_valid.groupby("연도")["평균기온"].mean().reset_index()

# 2. 데이터셋 분할
# 공통 테스트 데이터 (2006 ~ 2025)
test_df = annual_temp[
    (annual_temp["연도"] >= 2006) & (annual_temp["연도"] <= 2025)
]
X_test = test_df["연도"].values
y_test = test_df["평균기온"].values

# Model 1: 전체 데이터 (1908 ~ 2025)
X_full = annual_temp["연도"].values
y_full = annual_temp["평균기온"].values

# Model 2: 최근 50년 학습 (1956 ~ 2005)
train_50 = annual_temp[
    (annual_temp["연도"] >= 1956) & (annual_temp["연도"] <= 2005)
]
X_train_50 = train_50["연도"].values
y_train_50 = train_50["평균기온"].values

# Model 3: 최근 100년 학습 (1906 ~ 2005)
train_100 = annual_temp[
    (annual_temp["연도"] >= 1906) & (annual_temp["연도"] <= 2005)
]
X_train_100 = train_100["연도"].values
y_train_100 = train_100["평균기온"].values


# 3. 모델 학습 및 평가 함수
def evaluate_model(X_tr, y_tr, X_te, y_te, model_name):
    # 회귀계수 추정 (y = slope * X + intercept)
    slope, intercept = np.polyfit(X_tr, y_tr, 1)

    # 테스트 데이터 예측
    y_pred = slope * X_te + intercept

    # 평가지표 계산
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    return {
        "모델": model_name,
        "기울기 (℃/년)": slope,
        "절편": intercept,
        "MAE": mae,
        "MSE": mse,
        "R²": r2,
    }


# 결과 비교
results = []

# 전체 데이터 자체 평가
slope_f, intercept_f = np.polyfit(X_full, y_full, 1)
y_pred_f = slope_f * X_full + intercept_f
results.append(
    {
        "모델": "전체 데이터 (자체 평가)",
        "기울기 (℃/년)": slope_f,
        "절편": intercept_f,
        "MAE": mean_absolute_error(y_full, y_pred_f),
        "MSE": mean_squared_error(y_full, y_pred_f),
        "R²": r2_score(y_full, y_pred_f),
    }
)

# 50년 학습 -> 테스트 20년 평가
results.append(
    evaluate_model(
        X_train_50, y_train_50, X_test, y_test, "최근 50년 학습 (1956~2005)"
    )
)

# 100년 학습 -> 테스트 20년 평가
results.append(
    evaluate_model(
        X_train_100, y_train_100, X_test, y_test, "최근 100년 학습 (1906~2005)"
    )
)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
