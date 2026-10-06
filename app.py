import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("📊 기온 예측기")

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
    
    # 조건 적용 (2025년 이하 & 관측일 300일 이상)
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()
    
    # 1908년부터 지난 연수 계산
    filtered_df["지난연수"] = filtered_df["연도"] - 1908
    
    return filtered_df

df_total = load_data()

# 분석 기준 정보 계산
total_count = len(df_total)
start_year = int(df_total["연도"].min())
end_year = int(df_total["연도"].max())

# 전체 기간 회귀 모델 (1908년 기준)
X_tot = df_total["지난연수"].values
y_tot = df_total["연평균기온"].values
slope_tot, intercept_tot = np.polyfit(X_tot, y_tot, 1)

# 상관계수 계산
corr_matrix = np.corrcoef(df_total["연도"], y_tot)
correlation = corr_matrix[0, 1]

# 최근 20년 회귀 모델
df_recent20 = df_total[df_total["연도"] >= (end_year - 19)].copy()
X_rec = df_recent20["지난연수"].values
y_rec = df_recent20["연평균기온"].values
slope_rec, intercept_rec = np.polyfit(X_rec, y_rec, 1)

# 100년당 기온 상승량 계산
slope_tot_100y = slope_tot * 100
slope_rec_100y = slope_rec * 100

# 2. 메트릭 요약 정보 표시
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("분석 대상 해의 개수", f"{total_count}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{correlation:.4f}")
col5.metric("100년당 상승량 (전체)", f"{slope_tot_100y:+.2f} °C")

st.markdown("---")

# 3. 그래프 영역: 전체 기간과 최근 20년 회귀선 비교
st.subheader("📊 전체 기간과 최근 20년 회귀선 비교")

# X축 데이터 정의
x_tot_range = np.linspace(df_total["연도"].min(), df_total["연도"].max(), 100)
y_tot_line = slope_tot * (x_tot_range - 1908) + intercept_tot

x_rec_range = np.linspace(df_recent20["연도"].min(), df_recent20["연도"].max(), 50)
y_rec_line = slope_rec * (x_rec_range - 1908) + intercept_rec

fig = go.Figure()

# 실제 연평균기온 (파란 점)
fig.add_trace(
    go.Scatter(
        x=df_total["연도"],
        y=df_total["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(color="#1f77b4", size=6),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
    )
)

# 전체 기간 회귀선 (연한 파란선)
fig.add_trace(
    go.Scatter(
        x=x_tot_range,
        y=y_tot_line,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(color="#6baed6", width=2),
        hovertemplate="%{x:.0f}년 예측: %{y:.2f}°C<extra></extra>"
    )
)

# 최근 20년 회귀선 (빨간선)
fig.add_trace(
    go.Scatter(
        x=x_rec_range,
        y=y_rec_line,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(color="#d62728", width=2.5),
        hovertemplate="%{x:.0f}년 예측: %{y:.2f}°C<extra></extra>"
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=500,
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.8)"
    ),
    margin=dict(l=40, r=40, t=20, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 4. 연도별 기온 예측 슬라이더 및 수치 표시 영역
st.subheader("🔮 연도별 기온 예측")
st.caption("예측할 연도를 선택하세요.")

target_year = st.slider(
    "예측할 연도 선택",
    min_value=1900,
    max_value=2100,
    value=2055,
    step=1,
    label_visibility="collapsed"
)

# 전체 기간 회귀 모델 기준 예측값 계산
predicted_temp = slope_tot * (target_year - 1908) + intercept_tot

st.write(f"{target_year}년 예상 연평균기온")
st.markdown(f"<h1 style='color: #2c3e50; font-size: 2.5rem; margin-top: -10px;'>{predicted_temp:.2f} °C</h1>", unsafe_allow_html=True)
