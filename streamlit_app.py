import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import pickle
import __main__ 

# 1. 核心修复：挂载类以修复 Pickle 加载问题
from code_v1 import VentilationEnvironment, SarsaAgent, EpsilonGreedy, Softmax, UCB, RandomExploration
__main__.SarsaAgent = SarsaAgent
__main__.VentilationEnvironment = VentilationEnvironment
__main__.EpsilonGreedy = EpsilonGreedy

st.set_page_config(page_title="Smart Ventilation Sarsa Control", page_icon="🌬️", layout="wide")

# --- 最终版 CSS：强化图表标题、强制白底、深色数值 ---
st.markdown("""
<style>
    /* 1. 全局背景 */
    .stApp { background-color: ##0E1117 !important; }

    /* 2. Metric 卡片样式 */
    .big-metric { 
        font-size: 2.5rem; font-weight: 800; text-align: center; 
        color: #1E1E1E !important; margin: 0.2rem 0; 
    }
    .metric-box { 
        background-color: #ffffff !important; padding: 1.2rem; border-radius: 0.8rem; 
        margin: 0.5rem 0; text-align: center; border: 2px solid #E0E0E0 !important; 
        min-height: 140px; display: flex; flex-direction: column; justify-content: center;
    }
    .status-good-border { border-color: #28a745 !important; border-width: 3px !important; }
    .status-danger-border { border-color: #dc3545 !important; border-width: 3px !important; }
    .metric-label { font-size: 0.95rem; font-weight: 600; color: #555555 !important; }
    .metric-unit { font-size: 0.8rem; color: #888888 !important; }

    /* 3. 图表标题放大设置 */
    .chart-title {
        font-size: 1.4rem !important; /* 放大标题字号 */
        font-weight: 700 !important;   /* 加粗 */
        color: #ffffff !important;     /* 亮白色以便在深色背景下阅读 */
        margin-bottom: 8px !important;
        display: block;
    }

    /* 4. 强制图表内部 Canvas 完全变白 */
    [data-testid="stVegaLiteChart"] {
        padding: 10px !important;
        border-radius: 12px !important;
    }
    [data-testid="stVegaLiteChart"] svg { background-color: white !important; }

    /* 强制坐标轴文字和标签为深灰色 */
    [data-testid="stVegaLiteChart"] g.mark-text text, 
    [data-testid="stVegaLiteChart"] g.role-axis text,
    [data-testid="stVegaLiteChart"] g.role-legend text {
        fill: #31333F !important;
        font-size: 11px !important;
    }
    [data-testid="stVegaLiteChart"] g.role-axis path,
    [data-testid="stVegaLiteChart"] g.role-axis line {
        stroke: #D0D0D0 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 资源加载 ---
@st.cache_resource
def load_pretrained_agent():
    model_path = "agent.pkl" 
    if not os.path.exists(model_path): return None, "Model not found"
    try:
        with open(model_path, 'rb') as f: return pickle.load(f), None 
    except Exception as e: return None, str(e)

@st.cache_resource
def create_environment():
    try: return VentilationEnvironment(max_steps=96, discrete_actions_count=5), None
    except Exception as e: return None, str(e)

# --- 初始化 Session State ---
if 'initialized' not in st.session_state:
    agent, _ = load_pretrained_agent()
    env, _ = create_environment()
    st.session_state.agent = agent
    st.session_state.env = env
    st.session_state.current_step = 0
    st.session_state.cum_reward = 0.0
    st.session_state.simulation_running = False
    st.session_state.last_obs = [406.0, 0.0, 0.0, 0.0]
    st.session_state.last_info = {'energy_consumed': 0.0}
    st.session_state.history_df = pd.DataFrame(columns=[
        "Time", "CO2", "Setpoint", "Fan", "Energy", "People", "Step Reward", "Total Reward"
    ]).set_index("Time")
    st.session_state.initialized = True

# --- 侧边栏 ---
st.sidebar.title("🎮 Control Panel")
if st.sidebar.button("▶️ Start / Resume", use_container_width=True):
    st.session_state.simulation_running = True
if st.sidebar.button("⏸️ Pause", use_container_width=True):
    st.session_state.simulation_running = False
if st.sidebar.button("🔄 Reset", use_container_width=True):
    st.session_state.current_step = 0
    st.session_state.cum_reward = 0.0
    st.session_state.simulation_running = False
    st.session_state.last_obs = [406.0, 0.0, 0.0, 0.0]
    st.session_state.last_info = {'energy_consumed': 0.0}
    st.session_state.history_df = st.session_state.history_df.iloc[0:0] 
    st.rerun()

sim_speed = st.sidebar.slider("Speed", 0.5, 10.0, 2.0)

# --- 布局：Metrics ---
st.markdown("### 📊 Live Performance")
m_cols = st.columns(5)
metric_placeholders = [col.empty() for col in m_cols]

def render_metrics(obs, info, total_reward, env):
    co2_status = "status-good-border" if obs[0] < env.co2_setpoint else "status-danger-border"
    data = [
        ("CO₂", f"{obs[0]:.0f}", "ppm", co2_status),
        ("Fan Speed", f"{obs[1]*100:.0f}", "%", ""),
        ("People", f"{int(obs[3])}", "person", ""),
        ("Energy", f"{info['energy_consumed']:.1f}", "Wh", ""),
        ("Total Reward", f"{total_reward:.1f}", "pts", "")
    ]
    for i, (label, val, unit, border) in enumerate(data):
        metric_placeholders[i].markdown(
            f'<div class="metric-box {border}"><div class="metric-label">{label}</div>'
            f'<div class="big-metric">{val}</div><div class="metric-unit">{unit}</div></div>', 
            unsafe_allow_html=True
        )

# --- 布局：Charts (标题放大处理) ---
st.markdown("---")
h_df = st.session_state.history_df

col_l, col_r = st.columns(2)
with col_l:
    st.markdown('<span class="chart-title">CO2 Level (ppm)</span>', unsafe_allow_html=True)
    co2_chart = st.line_chart(h_df[["CO2", "Setpoint"]])
with col_r:
    st.markdown('<span class="chart-title">Fan Speed (%)</span>', unsafe_allow_html=True)
    fan_chart = st.line_chart(h_df[["Fan"]])

col_l2, col_r2 = st.columns(2)
with col_l2:
    st.markdown('<span class="chart-title">Energy Consumption (Wh)</span>', unsafe_allow_html=True)
    energy_chart = st.line_chart(h_df[["Energy"]])
with col_r2:
    st.markdown('<span class="chart-title">Occupancy (People)</span>', unsafe_allow_html=True)
    occ_chart = st.line_chart(h_df[["People"]])

col_l3, col_r3 = st.columns(2)
with col_l3:
    st.markdown('<span class="chart-title">Step Reward</span>', unsafe_allow_html=True)
    step_rew_chart = st.line_chart(h_df[["Step Reward"]])
with col_r3:
    st.markdown('<span class="chart-title">Cumulative Reward</span>', unsafe_allow_html=True)
    cum_rew_chart = st.line_chart(h_df[["Total Reward"]])

render_metrics(st.session_state.last_obs, st.session_state.last_info, st.session_state.cum_reward, st.session_state.env)

# --- 模拟循环 ---
if st.session_state.simulation_running:
    env, agent = st.session_state.env, st.session_state.agent
    if st.session_state.current_step == 0:
        obs, _ = env.reset()
        st.session_state.last_obs, st.session_state.state = obs, agent.discretize_state(obs)

    while st.session_state.simulation_running and st.session_state.current_step < env.max_steps:
        action = agent.select_action(st.session_state.state, training=False)
        obs, reward, term, trun, info = env.step(action)
        
        st.session_state.last_obs, st.session_state.last_info = obs, info
        st.session_state.state = agent.discretize_state(obs)
        st.session_state.current_step += 1
        st.session_state.cum_reward += reward
        
        cur_time = st.session_state.current_step * env.dt / 60.0
        new_row = pd.DataFrame({
            "CO2": [obs[0]], "Setpoint": [env.co2_setpoint], "Fan": [obs[1]*100],
            "Energy": [info['energy_consumed']], "People": [obs[3]], 
            "Step Reward": [reward], "Total Reward": [st.session_state.cum_reward]
        }, index=[cur_time])
        
        st.session_state.history_df = pd.concat([st.session_state.history_df, new_row])
        render_metrics(obs, info, st.session_state.cum_reward, env)
        
        co2_chart.add_rows(new_row[["CO2", "Setpoint"]])
        fan_chart.add_rows(new_row[["Fan"]])
        energy_chart.add_rows(new_row[["Energy"]])
        occ_chart.add_rows(new_row[["People"]])
        step_rew_chart.add_rows(new_row[["Step Reward"]])
        cum_rew_chart.add_rows(new_row[["Total Reward"]])

        if term or trun:
            st.session_state.simulation_running = False
            st.balloons()
            break
        time.sleep(0.1 / sim_speed)