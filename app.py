import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 예측 및 모델 평가", layout="wide")

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 평가")

# 데이터 로드 및 전처리
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

df_total = load_data()

# 데이터 분할
df_test = df_total[(df_total["연도"] >= 2006) & (df_total["연도"] <= 2025)].copy()
df_train_50 = df_total[(df_total["연도"] >= 1956) & (df_total["연도"] <= 2005)].copy()
df_train_100 = df_total[(df_total["연도"] >= 1906) & (df_total["연도"] <= 2005)].copy()


# =========================================================
# ① 훈련 데이터와 테스트 데이터
# =========================================================
st.subheader("① 훈련 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="테스트 데이터",
        value=f"{len(df_test)}개년",
        delta="2006~2025년",
        delta_color="off"
    )

with col2:
    st.metric(
        label="50년 학습 데이터",
        value=f"{len(df_train_50)}개년",
        delta="1956~2005년 중 조건을 만족한 자료",
        delta_color="off"
    )

with col3:
    st.metric(
        label="100년 학습 데이터",
        value=f"{len(df_train_100)}개년",
        delta="1906~2005년 중 조건을 만족한 자료",
        delta_color="off"
    )

st.info(
    "두 모델 모두 같은 최근 20년(2006~2025년)을 테스트 데이터로 사용합니다. "
    "따라서 50년을 학습한 모델과 100년을 학습한 모델의 예측 성능을 공정하게 비교할 수 있습니다."
)

st.markdown("---")


# =========================================================
# ② 전체 데이터로 만든 회귀모델
# =========================================================
st.subheader("② 전체 데이터로 만든 회귀모델")

X_all = df_total["지난연수"].values
y_all = df_total["연평균기온"].values

slope_all, intercept_all = np.polyfit(X_all, y_all, 1)
y_pred_all = slope_all * X_all + intercept_all

slope_100y_all = slope_all * 100
corr_all = np.corrcoef(df_total["연도"], y_all)[0, 1]
mae_all = mean_absolute_error(y_all, y_pred_all)
mse_all = mean_squared_error(y_all, y_pred_all)
r2_all = r2_score(y_all, y_pred_all)

m1, m2, m3, m4 = st.columns(4)
m1.metric("100년당 기온 변화", f"{slope_100y_all:.2f} °C")
m2.metric("상관계수 r", f"{corr_all:.3f}")
m3.metric("MAE", f"{mae_all:.3f} °C")
m4.metric("R²", f"{r2_all:.3f}")

st.caption(f"MSE = {mse_all:.3f}")

st.warning(
    "이 평가는 전체 데이터를 이용해 회귀선을 만든 뒤 같은 데이터를 다시 평가한 결과입니다. "
    "따라서 새로운 데이터에 대한 실제 예측 성능을 평가한 것은 아닙니다."
)

x_range_all = np.linspace(df_total["연도"].min(), df_total["연도"].max(), 100)
y_range_all = slope_all * (x_range_all - 1908) + intercept_all

fig_all = go.Figure()

fig_all.add_trace(
    go.Scatter(
        x=df_total["연도"],
        y=df_total["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(color="#1f77b4", size=6),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
    )
)

fig_all.add_trace(
    go.Scatter(
        x=x_range_all,
        y=y_range_all,
        mode="lines",
        name="전체 데이터 회귀선",
        line=dict(color="#6baed6", width=2),
        hovertemplate="%{x:.0f}년: %{y:.2f}°C<extra></extra>"
    )
)

fig_all.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=450,
    legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top", bgcolor="rgba(255, 255, 255, 0.8)"),
    margin=dict(l=40, r=40, t=20, b=40)
)

st.plotly_chart(fig_all, use_container_width=True)

st.markdown("---")


# =========================================================
# ③ 50년 학습과 100년 학습 비교
# =========================================================
st.subheader("③ 50년 학습과 100년 학습 비교")

def eval_model(df_train, df_test):
    X_tr = df_train["지난연수"].values
    y_tr = df_train["연평균기온"].values
    slope, intercept = np.polyfit(X_tr, y_tr, 1)
    
    X_te = df_test["지난연수"].values
    y_te = df_test["연평균기온"].values
    y_pred = slope * X_te + intercept
    
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    
    return slope, intercept, mae, mse, r2

slope_50, intercept_50, mae_50, mse_50, r2_50 = eval_model(df_train_50, df_test)
slope_100, intercept_100, mae_100, mse_100, r2_100 = eval_model(df_train_100, df_test)

# 기간별 기울기 계산 (50년당 vs 100년당)
slope50_50 = slope_50 * 50
slope100_100 = slope_100 * 100

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.markdown("### 최근 50년 학습")
    st.caption("학습 기간: 1956년 ~ 2005년 | 테스트: 2006년 ~ 2025년")
    
    m1, m2 = st.columns(2)
    m1.metric("50년당 기온 변화", f"{slope50_50:.2f} °C")
    m2.metric("MAE", f"{mae_50:.3f} °C")
    
    m3, m4 = st.columns(2)
    m3.metric("MSE", f"{mse_50:.3f}")
    m4.metric("R²", f"{r2_50:.3f}")

with comp_col2:
    st.markdown("### 최근 100년 학습")
    st.caption("학습 기간: 1906년 ~ 2005년 | 테스트: 2006년 ~ 2025년")
    
    m1, m2 = st.columns(2)
    m1.metric("100년당 기온 변화", f"{slope100_100:.2f} °C")
    m2.metric("MAE", f"{mae_100:.3f} °C")
    
    m3, m4 = st.columns(2)
    m3.metric("MSE", f"{mse_100:.3f}")
    m4.metric("R²", f"{r2_100:.3f}")

x_range_comp = np.linspace(1900, 2025, 125)
y_line_50 = slope_50 * (x_range_comp - 1908) + intercept_50
y_line_100 = slope_100 * (x_range_comp - 1908) + intercept_100

fig_comp = go.Figure()

fig_comp.add_trace(
    go.Scatter(
        x=df_total["연도"],
        y=df_total["연평균기온"],
        mode="markers",
        name="전체 관측 데이터",
        marker=dict(color="#cbd5e1", size=5),
        hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>"
    )
)

fig_comp.add_trace(
    go.Scatter(
        x=df_test["연도"],
        y=df_test["연평균기온"],
        mode="markers",
        name="테스트 데이터 (2006~2025)",
        marker=dict(color="#0f172a", size=7),
        hovertemplate="%{x}년(테스트): %{y:.2f}°C<extra></extra>"
    )
)

fig_comp.add_trace(
    go.Scatter(
        x=x_range_comp,
        y=y_line_50,
        mode="lines",
        name="50년 학습 회귀선",
        line=dict(color="#f97316", width=2.5),
        hovertemplate="%{x:.0f}년 예측(50년): %{y:.2f}°C<extra></extra>"
    )
)

fig_comp.add_trace(
    go.Scatter(
        x=x_range_comp,
        y=y_line_100,
        mode="lines",
        name="100년 학습 회귀선",
        line=dict(color="#16a34a", width=2.5),
        hovertemplate="%{x:.0f}년 예측(100년): %{y:.2f}°C<extra></extra>"
    )
)

fig_comp.update_layout(
    title="학습 기간별 회귀선의 최근 20년 테스트 데이터 예측 성능 비교",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    template="plotly_white",
    height=480,
    legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", bgcolor="rgba(255, 255, 255, 0.8)")
)

st.plotly_chart(fig_comp, use_container_width=True)
