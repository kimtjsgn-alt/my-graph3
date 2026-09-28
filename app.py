import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("1908년부터 계산된 연수 기반 회귀 모델을 통해 연도별 예상 기온 및 기온 상승 추세를 분석합니다.")

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
    
    # 기본 필터링 조건: 2025년 이하 & 관측일 300일 이상
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

df_total = load_data()

# 최근 20년 데이터 필터링
max_year_data = df_total["연도"].max()
df_recent20 = df_total[df_total["연도"] >= (max_year_data - 19)].copy()

# 2. 회귀분석 및 상관계수 계산 함수
def fit_regression(df_input):
    X = df_input["지난연수"].values
    y = df_input["연평균기온"].values
    slope, intercept = np.polyfit(X, y, 1)
    
    corr_matrix = np.corrcoef(df_input["연도"], y)
    correlation = corr_matrix[0, 1]
    
    slope_100y = slope * 100  # 100년당 기온 변화량
    return slope, intercept, slope_100y, correlation

# 전체 및 최근 20년 모델 계산
slope_tot, intercept_tot, slope100_tot, corr_tot = fit_regression(df_total)
slope_rec, intercept_rec, slope100_rec, corr_rec = fit_regression(df_recent20)

# 3. 데이터 요약 및 기울기(100년당 기온 상승량) 비교 표시
st.subheader("📊 데이터 분석 요약 및 기온 상승률 비교")

m_col1, m_col2, m_col3 = st.columns(3)
min_year = int(df_total["연도"].min())
max_year = int(df_total["연도"].max())
total_years = len(df_total)

m_col1.metric("분석 대상 해의 개수", f"{total_years}개")
m_col2.metric("시작 연도", f"{min_year}년")
m_col3.metric("끝 연도", f"{max_year}년")

st.markdown("### 🔥 100년당 기온 상승량 비교")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.info("🌐 **전체 기간 모델**")
    st.metric(
        label="100년당 기온 상승량",
        value=f"{slope100_tot:+.2f} °C / 100년",
        delta=f"상관계수: {corr_tot:.4f}"
    )

with comp_col2:
    st.warning("⚡ **최근 20년 모델**")
    delta_val = slope100_rec - slope100_tot
    st.metric(
        label="100년당 기온 상승량",
        value=f"{slope100_rec:+.2f} °C / 100년",
        delta=f"전체 대비 {delta_val:+.2f} °C (상관계수: {corr_rec:.4f})"
    )

st.markdown("---")

# 4. 연도 선택 슬라이더 및 예측 비교
target_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2025, step=1)

years_passed = target_year - 1908
pred_tot = slope_tot * years_passed + intercept_tot
pred_rec = slope_rec * years_passed + intercept_rec

st.subheader(f"🎯 {target_year}년 예상 평균기온 비교")
p_col1, p_col2 = st.columns(2)

with p_col1:
    st.markdown(f"**[전체 기간 모델]**")
    st.markdown(f"# **{pred_tot:.2f} °C**")

with p_col2:
    st.markdown(f"**[최근 20년 모델]**")
    st.markdown(f"# **{pred_rec:.2f} °C**")

st.markdown("---")

# 5. Plotly 시각화 (두 회귀선 나란히 비교)
line_years = np.arange(1900, 2101)
line_x = line_years - 1908
line_y_tot = slope_tot * line_x + intercept_tot
line_y_rec = slope_rec * line_x + intercept_rec

fig = go.Figure()

# 관측 데이터
fig.add_trace(
    go.Scatter(
        x=df_total["연도"],
        y=df_total["연평균기온"],
        mode="markers",
        name="관측 데이터",
        marker=dict(color="#6c757d", size=7, opacity=0.7),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
    )
)

# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y_tot,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="%{x}년 전체모델 예측: %{y:.2f}°C<extra></extra>"
    )
)

# 최근 20년 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y_rec,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(color="#d62728", width=2.5, dash="dash"),
        hovertemplate="%{x}년 최근20년모델 예측: %{y:.2f}°C<extra></extra>"
    )
)

# 선택 연도 예측점 표시
fig.add_trace(
    go.Scatter(
        x=[target_year, target_year],
        y=[pred_tot, pred_rec],
        mode="markers+text",
        name="선택 연도 예측점",
        marker=dict(color=["#1f77b4", "#d62728"], size=12, symbol="star"),
        text=[f"{pred_tot:.2f}°C", f"{pred_rec:.2f}°C"],
        textposition="top center",
        hovertemplate="<b>%{x}년 예측: %{y:.2f}°C</b><extra></extra>"
    )
)

fig.update_layout(
    title="서울 연도별 평균기온 추세 및 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
