import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("1908년부터 계산된 연수 기반 회귀 모델을 통해 연도별 예상 기온을 예측합니다.")

# 1. 데이터 로드 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    yearly_summary = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

df = load_data()

# 2. 회귀분석 및 상관계수 계산
X = df["지난연수"].values
y = df["연평균기온"].values

slope, intercept = np.polyfit(X, y, 1)

corr_matrix = np.corrcoef(df["연도"], y)
correlation = corr_matrix[0, 1]

# 3. 데이터 요약정보 표시
min_year = int(df["연도"].min())
max_year = int(df["연도"].max())
total_years = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 해의 개수", f"{total_years}개")
col2.metric("시작 연도", f"{min_year}년")
col3.metric("끝 연도", f"{max_year}년")
col4.metric("상관계수", f"{correlation:.4f}")

st.markdown("---")

# 4. 연도 선택 및 예측 기온 표시
target_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)

years_passed = target_year - 1908
predicted_temp = slope * years_passed + intercept

st.subheader(f"🎯 {target_year}년 예상 평균기온")
st.markdown(f"# **{predicted_temp:.2f} °C**")

st.markdown("---")

# 5. Plotly 시각화 (산점도 + 회귀선)
line_years = np.arange(1900, 2101)
line_x = line_years - 1908
line_y = slope * line_x + intercept

fig = go.Figure()

# 관측 데이터
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["연평균기온"],
        mode="markers",
        name="관측 데이터",
        marker=dict(color="#1f77b4", size=8),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color="#ff7f0e", width=2),
        hovertemplate="%{x}년 예측: %{y:.2f}°C<extra></extra>"
    )
)

# 선택 연도 강조
fig.add_trace(
    go.Scatter(
        x=[target_year],
        y=[predicted_temp],
        mode="markers",
        name="선택한 연도 예측값",
        marker=dict(color="#d62728", size=14, symbol="star"),
        hovertemplate="<b>%{x}년 예측: %{y:.2f}°C</b><extra></extra>"
    )
)

# layout 수정 부분: orient -> orientation
fig.update_layout(
    title="서울 연도별 평균기온 및 회귀 직선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
