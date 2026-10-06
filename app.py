import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 예측 및 회귀 모델 평가", layout="wide")

st.title("📊 서울 연평균 기온 선형회귀 모델 분석 및 평가")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 관측일수 및 연평균기온 계산
    yearly_summary = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 필터링 조건: 2025년 이하 & 관측일 300일 이상
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    # 1908년부터 지난 연수 계산 (독립변수 X)
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

df_total = load_data()

# 2. 데이터 분할 (Train / Test)
# 공통 테스트 데이터: 최근 20년 (2006~2025)
df_test = df_total[(df_total["연도"] >= 2006) & (df_total["연도"] <= 2025)].copy()

# 훈련 데이터 A: 최근 50년 (1956~2005)
df_train_50 = df_total[(df_total["연도"] >= 1956) & (df_total["연도"] <= 2005)].copy()

# 훈련 데이터 B: 최근 100년 (1906~2005)
df_train_100 = df_total[(df_total["연도"] >= 1906) & (df_total["연도"] <= 2005)].copy()

# 회귀 모델 훈련 및 평가 함수
def train_and_eval(df_train, df_test):
    # 훈련
    X_train = df_train["지난연수"].values
    y_train = df_train["연평균기온"].values
    slope, intercept = np.polyfit(X_train, y_train, 1)
    
    # 테스트 데이터 예측
    X_test = df_test["지난연수"].values
    y_test = df_test["연평균기온"].values
    y_pred = slope * X_test + intercept
    
    # 성능 평가
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    slope_100y = slope * 100
    
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_100y": slope_100y,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "train_count": len(df_train)
    }

# 모델 훈련 및 평가
res_50 = train_and_eval(df_train_50, df_test)
res_100 = train_and_eval(df_train_100, df_test)

# 전체 데이터 회귀분석 (참고용)
X_all = df_total["지난연수"].values
y_all = df_total["연평균기온"].values
slope_all, intercept_all = np.polyfit(X_all, y_all, 1)

# 3. 모델 비교 결과 UI
st.subheader("🧪 최근 50년 vs 최근 100년 학습 모델 비교 (테스트: 최근 20년 2006~2025)")

col_a, col_b = st.columns(2)

with col_a:
    st.info("📌 **최근 50년 학습 모델 (1956~2005년)**")
    st.write(f"- 학습 데이터 수: `{res_50['train_count']}개`")
    st.write(f"- **100년당 기온 상승량**: **`{res_50['slope_100y']:+.2f} °C`**")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{res_50['mae']:.3f}")
    m2.metric("MSE", f"{res_50['mse']:.3f}")
    m3.metric("R²", f"{res_50['r2']:.3f}")

with col_b:
    st.success("📌 **최근 100년 학습 모델 (1906~2005년)**")
    st.write(f"- 학습 데이터 수: `{res_100['train_count']}개`")
    st.write(f"- **100년당 기온 상승량**: **`{res_100['slope_100y']:+.2f} °C`**")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{res_100['mae']:.3f}")
    m2.metric("MSE", f"{res_100['mse']:.3f}")
    m3.metric("R²", f"{res_100['r2']:.3f}")

# 수치 비교 표
st.markdown("##### 📋 성능 및 기울기 비교 요약표")
comparison_df = pd.DataFrame({
    "구분": ["최근 50년 학습 (1956~2005)", "최근 100년 학습 (1906~2005)"],
    "학습 데이터 개수": [res_50["train_count"], res_100["train_count"]],
    "100년당 상승량(°C)": [f"{res_50['slope_100y']:+.2f}", f"{res_100['slope_100y']:+.2f}"],
    "MAE (평균절대오차)": [f"{res_50['mae']:.4f}", f"{res_100['mae']:.4f}"],
    "MSE (평균제곱오차)": [f"{res_50['mse']:.4f}", f"{res_100['mse']:.4f}"],
    "R² (결정계수)": [f"{res_50['r2']:.4f}", f"{res_100['r2']:.4f}"]
})
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.markdown("---")

# 4. Plotly 시각화 (학습 기간별 회귀선 비교)
st.subheader("📈 학습 기간에 따른 회귀선 예측 추세 비교")

x_range = np.linspace(1900, 2030, 150)
x_years_passed = x_range - 1908

line_y_50 = res_50["slope"] * x_years_passed + res_50["intercept"]
line_y_100 = res_100["slope"] * x_years_passed + res_100["intercept"]
line_y_all = slope_all * x_years_passed + intercept_all

fig = go.Figure()

# 전체 실제 데이터 (배경)
fig.add_trace(
    go.Scatter(
        x=df_total["연도"],
        y=df_total["연평균기온"],
        mode="markers",
        name="과거 학습 데이터",
        marker=dict(color="#adc2d6", size=6),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
    )
)

# 테스트 데이터 (최근 20년 강조)
fig.add_trace(
    go.Scatter(
        x=df_test["연도"],
        y=df_test["연평균기온"],
        mode="markers",
        name="테스트 데이터 (2006~2025)",
        marker=dict(color="#1f77b4", size=8),
        hovertemplate="%{x}년(테스트): %{y:.2f}°C<extra></extra>"
    )
)

# 50년 학습 회귀선
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=line_y_50,
        mode="lines",
        name="50년 학습 회귀선",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        hovertemplate="%{x:.0f}년 예측(50년모델): %{y:.2f}°C<extra></extra>"
    )
)

# 100년 학습 회귀선
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=line_y_100,
        mode="lines",
        name="100년 학습 회귀선",
        line=dict(color="#2ca02c", width=2.5),
        hovertemplate="%{x:.0f}년 예측(100년모델): %{y:.2f}°C<extra></extra>"
    )
)

fig.update_layout(
    title="서울 연평균 기온 회귀선 및 테스트 데이터 예측 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=500,
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.8)"
    ),
    margin=dict(l=40, r=40, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 5. 연도 선택 및 예측 기온 표시
st.subheader("🔮 모델별 미래 연도 기온 예측")

target_year = st.slider("예측할 연도 선택", min_value=1900, max_value=2100, value=2030, step=1)
target_passed = target_year - 1908

pred_50 = res_50["slope"] * target_passed + res_50["intercept"]
pred_100 = res_100["slope"] * target_passed + res_100["intercept"]

p_col1, p_col2 = st.columns(2)
with p_col1:
    st.write(f"**[최근 50년 학습 모델]** {target_year}년 예상 기온")
    st.markdown(f"### **{pred_50:.2f} °C**")

with p_col2:
    st.write(f"**[최근 100년 학습 모델]** {target_year}년 예상 기온")
    st.markdown(f"### **{pred_100:.2f} °C**")
