# app.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="随机过程：布朗运动及其可视化模拟",
    layout="wide"
)

# =============================
# Utilities
# =============================
def _spd_corr_matrix(dim: int, rho: float) -> np.ndarray:
    rho = float(np.clip(rho, -0.99, 0.99))
    C = np.full((dim, dim), rho, dtype=float)
    np.fill_diagonal(C, 1.0)
    lower = -1.0 / (dim - 1) + 1e-6
    if rho <= lower:
        rho = lower
        C = np.full((dim, dim), rho, dtype=float)
        np.fill_diagonal(C, 1.0)
    return C


def _plot_3d_swarm(paths: np.ndarray, t_idx: int, n_traj: int, title: str) -> go.Figure:
    T, N, _ = paths.shape
    t_idx = int(np.clip(t_idx, 0, T - 1))

    pts = paths[t_idx]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.75),
            name=f"粒子位置（t={t_idx}）"
        )
    )

    n_traj = int(np.clip(n_traj, 0, min(N, 50)))
    if n_traj > 0 and t_idx >= 1:
        pick = np.linspace(0, N - 1, n_traj, dtype=int)
        for k, pid in enumerate(pick):
            tr = paths[:t_idx + 1, pid, :]
            fig.add_trace(
                go.Scatter3d(
                    x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
                    mode="lines",
                    name=f"轨迹 {pid}",
                    showlegend=(k < 5),
                    line=dict(width=3),
                    opacity=0.9
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig


def _line_plot(series: np.ndarray, title: str, yname: str) -> go.Figure:
    x = np.arange(len(series))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=series, mode="lines", name=yname))
    fig.update_layout(
        title=title,
        xaxis_title="step",
        yaxis_title=yname,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# =============================
# Simulators
# =============================
@st.cache_data(show_spinner=False)
def sim_standard_bm_3d(n_particles, steps, dt, sigma, seed):
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(dt), size=(steps, n_particles, 3))
    X = np.zeros((steps + 1, n_particles, 3))
    X[1:] = sigma * np.cumsum(dW, axis=0)
    return X


@st.cache_data(show_spinner=False)
def sim_drift_bm_3d(n_particles, steps, dt, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(dt), size=(steps, n_particles, 3))
    X = np.zeros((steps + 1, n_particles, 3))
    X[1:] = np.cumsum(mu * dt + sigma * dW, axis=0)
    return X


@st.cache_data(show_spinner=False)
def sim_correlated_bm_3d(n_particles, steps, dt, sigma, rho, seed):
    rng = np.random.default_rng(seed)
    C = _spd_corr_matrix(3, rho)
    L = np.linalg.cholesky(C)

    Z = rng.normal(size=(steps, n_particles, 3))
    dB = np.sqrt(dt) * (Z @ L.T)

    X = np.zeros((steps + 1, n_particles, 3))
    X[1:] = sigma * np.cumsum(dB, axis=0)
    return X


@st.cache_data(show_spinner=False)
def sim_gbm_3d(n_particles, steps, dt, mu, sigma, rho, x0, seed):
    rng = np.random.default_rng(seed)
    C = _spd_corr_matrix(3, rho)
    L = np.linalg.cholesky(C)

    Z = rng.normal(size=(steps, n_particles, 3))
    dB = np.sqrt(dt) * (Z @ L.T)

    logX = np.zeros((steps + 1, n_particles, 3))
    logX[0] = np.log(max(x0, 1e-9))
    logX[1:] = np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * dB, axis=0) + logX[0]

    return np.exp(logX)


# =============================
# Sidebar
# =============================
st.title("随机过程：布朗运动及其可视化模拟")

with st.sidebar:
    st.header("全局参数")
    steps = st.slider("步数", 50, 600, 240, 10)
    dt = st.slider("时间步长 Δt", 0.001, 0.05, 0.01, 0.001)
    n_particles = st.slider("粒子数", 50, 2000, 500, 50)
    show_traj = st.slider("轨迹条数", 0, 30, 8, 1)
    seed = st.number_input("随机种子", 0, 10_000_000, 7)

    st.divider()
    st.subheader("模型参数")
    sigma = st.slider("σ", 0.1, 3.0, 1.0, 0.05)
    mu = st.slider("μ", -2.0, 2.0, 0.3, 0.05)
    rho = st.slider("ρ", -0.9, 0.9, 0.5, 0.05)
    x0 = st.slider("GBM 初值 X(0)", 0.1, 10.0, 1.0, 0.1)

tab1, tab2, tab3, tab4 = st.tabs(
    ["标准布朗运动", "漂移布朗运动", "相关布朗运动", "几何布朗运动"]
)

# =============================
# Tab 1
# =============================
with tab1:
    st.subheader("标准布朗运动（维纳过程）")
    st.latex(r"B(0)=0,\quad B(t)-B(s)\sim\mathcal N(0,t-s)")
    st.markdown(
        r"- **直观解释**：随机微小扰动不断累积，形成连续但不可导的轨迹。" "\n"
        r"- **扩散特性**：方差随时间线性增长。"
    )

    X = sim_standard_bm_3d(n_particles, steps, dt, sigma, seed)
    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t1")
    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, "标准布朗运动：三维粒子群"),
        width="stretch"
    )

# =============================
# Tab 2
# =============================
with tab2:
    st.subheader("漂移布朗运动")
    st.latex(r"X(t)=\mu t+\sigma B(t)")
    st.markdown(
        r"- **直观解释**：随机抖动上叠加确定性线性趋势。" "\n"
        r"- $\mu>0$ 向上漂移，$\mu<0$ 向下漂移。"
    )

    X = sim_drift_bm_3d(n_particles, steps, dt, mu, sigma, seed + 1)
    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t2")
    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, "漂移布朗运动：三维粒子群"),
        width="stretch"
    )

# =============================
# Tab 3
# =============================
with tab3:
    st.subheader("相关布朗运动")
    st.latex(r"\mathrm{Cov}(B_i,B_j)=\rho_{ij}t")
    st.markdown(
        r"- **直观解释**：不同方向的随机扰动不再独立，而是具有相关性。" "\n"
        r"- $\rho>0$ 同涨同跌，$\rho<0$ 相互牵制。"
    )

    X = sim_correlated_bm_3d(n_particles, steps, dt, sigma, rho, seed + 2)
    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t3")
    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, f"相关布朗运动（ρ={rho:.2f}）"),
        width="stretch"
    )

# =============================
# Tab 4
# =============================
with tab4:
    st.subheader("几何布朗运动（GBM）")
    st.latex(r"\mathrm dX=\mu X\,\mathrm dt+\sigma X\,\mathrm dB")
    st.markdown(
        r"- **直观解释**：随机扰动按比例作用，状态始终保持正值。" "\n"
        r"- 常用于描述价格、规模等指数型随机演化。"
    )

    X = sim_gbm_3d(n_particles, steps, dt, mu, sigma, rho, x0, seed + 3)
    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t4")
    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, "几何布朗运动：三维粒子群"),
        width="stretch"
    )

st.divider()
st.markdown(
    r"**说明**：左侧调整参数 $\mu,\sigma,\rho,\Delta t$，观察不同布朗运动的随机行为差异。"
)
