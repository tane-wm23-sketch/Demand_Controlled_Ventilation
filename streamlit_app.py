import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from collections import deque, defaultdict
import pickle

# Import your environment and agent classes
# Update these imports with your actual file names
from code_v1 import VentilationEnvironment
from code_v1 import DynaQAgent

st.set_page_config(
    page_title="Smart Ventilation Control",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .status-good {
        color: #28a745;
    }
    .status-warning {
        color: #ffc107;
    }
    .status-danger {
        color: #dc3545;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load pre-trained agent (cached)
@st.cache_resource
def load_pretrained_agent():
    """Load your pre-trained agent"""
    try:
        # Update with your model path
        model_path = "agent.pkl"  # Change this to your model filename
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruct agent with proper parameters
        agent = DynaQAgent(
            n_actions=5,
            learning_rate=0.1,
            discount_factor=0.99,
            planning_steps=5,
            state_discretization=(20, 5, 2, 5),
            exploration_strategy="softmax",
            temperature=1.0,
            temp_min=0.01,
            temp_decay=0.999,
            use_replay=True,
            replay_buffer_size=10000,
            replay_batch_size=32
        )
        
        # Load the saved Q-table and model
        agent.q_table = defaultdict(lambda: np.zeros(agent.n_actions), data['q_table'])
        agent.model = data['model']
        agent.episode_rewards = data.get('episode_rewards', [])
        agent.episode_steps = data.get('episode_steps', [])
        
        return agent, None
    except FileNotFoundError:
        return None, "Model file not found. Please ensure the model file is in the same directory."
    except Exception as e:
        return None, f"Error loading agent: {str(e)}"

# Create environment (cached)
@st.cache_resource
def create_environment():
    """Create environment with fixed parameters"""
    try:
        env = VentilationEnvironment(
            max_steps=96,
            dt=5.0,
            room_area=50.0,
            room_height=3.0,
            co2_setpoint=1500.0,
            co2_outdoor=406.0,
            k_iaq=10.0,
            k_energy=0.0002,
            k_switch=10.0,
            discrete_actions_count=5
        )
        return env, None
    except Exception as e:
        return None, f"Error creating environment: {str(e)}"

# Initialize session state
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {
        'co2_history': deque(maxlen=200),
        'fan_speed_history': deque(maxlen=200),
        'occupancy_history': deque(maxlen=200),
        'reward_history': deque(maxlen=200),
        'energy_history': deque(maxlen=200),
        'time_steps': deque(maxlen=200)
    }
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'env' not in st.session_state:
    st.session_state.env = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'observation' not in st.session_state:
    st.session_state.observation = None
if 'state' not in st.session_state:
    st.session_state.state = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def reset_simulation():
    """Reset simulation to initial state"""
    if st.session_state.env is not None:
        observation, info = st.session_state.env.reset()
        st.session_state.observation = observation
        if st.session_state.agent is not None:
            st.session_state.state = st.session_state.agent.discretize_state(observation)
    
    # Clear history
    for key in st.session_state.simulation_data:
        st.session_state.simulation_data[key].clear()
    
    st.session_state.current_step = 0

def create_co2_chart(time_data, co2_data, setpoint, outdoor):
    """Create CO2 chart using Plotly"""
    fig = go.Figure()
    
    # CO2 line
    fig.add_trace(go.Scatter(
        x=list(time_data),
        y=list(co2_data),
        mode='lines',
        name='CO₂ Level',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Setpoint line
    fig.add_trace(go.Scatter(
        x=list(time_data),
        y=[setpoint] * len(time_data),
        mode='lines',
        name='Setpoint',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Outdoor line
    fig.add_trace(go.Scatter(
        x=list(time_data),
        y=[outdoor] * len(time_data),
        mode='lines',
        name='Outdoor',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Indoor CO₂ Concentration',
        xaxis_title='Time (hours)',
        yaxis_title='CO₂ (ppm)',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    
    return fig

def create_fan_chart(time_data, fan_data):
    """Create Fan Speed chart using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(time_data),
        y=list(fan_data),
        mode='lines',
        name='Fan Speed',
        fill='tozeroy',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title='Ventilation Fan Speed',
        xaxis_title='Time (hours)',
        yaxis_title='Fan Speed (%)',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified',
        yaxis=dict(range=[0, 105])
    )
    
    return fig

def create_combined_chart(time_data, energy_data, occupancy_data, max_occ):
    """Create combined Energy and Occupancy chart using Plotly"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Energy line
    fig.add_trace(
        go.Scatter(
            x=list(time_data),
            y=list(energy_data),
            mode='lines',
            name='Energy',
            line=dict(color='#2ca02c', width=2)
        ),
        secondary_y=False
    )
    
    # Occupancy line
    fig.add_trace(
        go.Scatter(
            x=list(time_data),
            y=list(occupancy_data),
            mode='lines',
            name='Occupancy',
            fill='tozeroy',
            line=dict(color='#d62728', width=2),
            opacity=0.7
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Energy Consumption & Room Occupancy',
        xaxis_title='Time (hours)',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Energy (Wh)", secondary_y=False)
    fig.update_yaxes(title_text="Occupancy (people)", secondary_y=True, range=[0, max_occ + 1])
    
    return fig

# Load agent and environment on startup
if not st.session_state.initialized:
    agent, agent_error = load_pretrained_agent()
    env, env_error = create_environment()
    
    if agent_error:
        st.error(f"❌ Agent Loading Error: {agent_error}")
    if env_error:
        st.error(f"❌ Environment Error: {env_error}")
    
    if agent is not None and env is not None:
        st.session_state.agent = agent
        st.session_state.env = env
        reset_simulation()
        st.session_state.initialized = True

# Main App
st.title("🌬️ Smart Ventilation Control System - Live Demo")

# Check if system is ready
if st.session_state.agent is None or st.session_state.env is None:
    st.error("""
    ❌ **System Not Ready**
    
    Please ensure:
    1. Your trained agent file exists (default: `dynaq_softmax_agent.pkl`)
    2. VentilationEnvironment and DynaQAgent classes are properly imported
    3. All dependencies are installed
    """)
    st.stop()

# Sidebar - Controls
st.sidebar.title("🎮 Control Panel")

# System Status
st.sidebar.markdown("### 📊 System Status")
st.sidebar.success("✅ Agent Loaded")
st.sidebar.success("✅ Environment Ready")
st.sidebar.info(f"🏢 Room: {st.session_state.env.room_area}m² × {st.session_state.env.room_height}m")
st.sidebar.info(f"👥 Max Occupancy: {st.session_state.env.max_occupancy} people")

st.sidebar.markdown("---")

# Occupancy Control
st.sidebar.markdown("### 👥 Occupancy Control")

max_occupancy = st.session_state.env.max_occupancy

manual_occupancy = st.sidebar.slider(
    "Number of People",
    min_value=0,
    max_value=max_occupancy,
    value=0,
    step=1,
    help=f"Set room occupancy (Max: {max_occupancy})"
)

use_manual_occupancy = st.sidebar.checkbox(
    "Override Occupancy",
    value=False,
    help="Check to manually control occupancy instead of using automatic meeting schedule"
)

if use_manual_occupancy:
    st.sidebar.info(f"🔧 Manual Mode: {manual_occupancy} people")
else:
    st.sidebar.info("🤖 Automatic Mode: Using meeting schedule")

st.sidebar.markdown("---")

# Simulation Controls
st.sidebar.markdown("### ⚙️ Simulation Controls")

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("▶️ Start", use_container_width=True):
        st.session_state.simulation_running = True

with col2:
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.simulation_running = False

if st.sidebar.button("🔄 Reset", use_container_width=True):
    reset_simulation()
    st.rerun()

# Speed control
simulation_speed = st.sidebar.slider(
    "Simulation Speed",
    min_value=0.1,
    max_value=3.0,
    value=1.0,
    step=0.1,
    help="Controls how fast the simulation runs"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.caption(f"Agent: {st.session_state.agent.exploration_strategy.name}")
st.sidebar.caption(f"Q-table size: {len(st.session_state.agent.q_table)}")
st.sidebar.caption(f"Model size: {len(st.session_state.agent.model)}")

# Main Content Area
metrics_container = st.container()
charts_container = st.container()
details_container = st.container()

# Metrics Display
with metrics_container:
    st.markdown("### 📊 Real-Time Metrics")
    
    metric_cols = st.columns(5)
    
    # Placeholders for metrics
    co2_placeholder = metric_cols[0].empty()
    fan_placeholder = metric_cols[1].empty()
    occupancy_placeholder = metric_cols[2].empty()
    energy_placeholder = metric_cols[3].empty()
    reward_placeholder = metric_cols[4].empty()

# Charts Display
with charts_container:
    st.markdown("### 📈 Live Charts")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        co2_chart_placeholder = st.empty()
    
    with chart_col2:
        fan_chart_placeholder = st.empty()
    
    energy_chart_placeholder = st.empty()

# Detailed Information
with details_container:
    st.markdown("### 📋 Episode Information")
    
    detail_cols = st.columns(4)
    
    step_placeholder = detail_cols[0].empty()
    time_placeholder = detail_cols[1].empty()
    violations_placeholder = detail_cols[2].empty()
    compliance_placeholder = detail_cols[3].empty()

# Simulation Loop
if st.session_state.simulation_running:
    
    env = st.session_state.env
    agent = st.session_state.agent
    
    # Get current state
    if st.session_state.observation is None:
        observation, _ = env.reset()
        st.session_state.observation = observation
        st.session_state.state = agent.discretize_state(observation)
    
    # Override occupancy if manual control is enabled
    if use_manual_occupancy:
        env.occupancy = manual_occupancy
        env.light_level = 1.0 if manual_occupancy > 0 else 0.0
        st.session_state.observation[3] = float(manual_occupancy)
        st.session_state.state = agent.discretize_state(st.session_state.observation)
    
    # Agent selects action
    action = agent.select_action(st.session_state.state, training=False)
    
    # Environment step
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Override occupancy again after step if manual control
    if use_manual_occupancy:
        env.occupancy = manual_occupancy
        env.light_level = 1.0 if manual_occupancy > 0 else 0.0
        observation[3] = float(manual_occupancy)
    
    st.session_state.observation = observation
    st.session_state.state = agent.discretize_state(observation)
    
    # Store data
    current_time = st.session_state.current_step * env.dt / 60.0  # hours
    st.session_state.simulation_data['time_steps'].append(current_time)
    st.session_state.simulation_data['co2_history'].append(observation[0])
    st.session_state.simulation_data['fan_speed_history'].append(observation[1] * 100)
    st.session_state.simulation_data['occupancy_history'].append(observation[3])
    st.session_state.simulation_data['reward_history'].append(reward)
    st.session_state.simulation_data['energy_history'].append(info['energy_consumed'])
    
    st.session_state.current_step += 1
    
    # Update Metrics
    co2_level = observation[0]
    co2_status = "status-good" if co2_level < env.co2_setpoint else "status-danger"
    
    co2_placeholder.markdown(f"""
    <div class="metric-box">
        <div style="color: #666; font-size: 0.9rem;">CO₂ Level</div>
        <div class="big-metric {co2_status}">{co2_level:.0f}</div>
        <div style="color: #666; font-size: 0.8rem;">ppm</div>
    </div>
    """, unsafe_allow_html=True)
    
    fan_placeholder.markdown(f"""
    <div class="metric-box">
        <div style="color: #666; font-size: 0.9rem;">Fan Speed</div>
        <div class="big-metric">{observation[1]*100:.0f}</div>
        <div style="color: #666; font-size: 0.8rem;">%</div>
    </div>
    """, unsafe_allow_html=True)
    
    occupancy_placeholder.markdown(f"""
    <div class="metric-box">
        <div style="color: #666; font-size: 0.9rem;">Occupancy</div>
        <div class="big-metric">{int(observation[3])}</div>
        <div style="color: #666; font-size: 0.8rem;">people</div>
    </div>
    """, unsafe_allow_html=True)
    
    energy_placeholder.markdown(f"""
    <div class="metric-box">
        <div style="color: #666; font-size: 0.9rem;">Energy Used</div>
        <div class="big-metric">{info['energy_consumed']:.1f}</div>
        <div style="color: #666; font-size: 0.8rem;">Wh</div>
    </div>
    """, unsafe_allow_html=True)
    
    reward_color = "status-good" if reward > 0 else "status-danger"
    reward_placeholder.markdown(f"""
    <div class="metric-box">
        <div style="color: #666; font-size: 0.9rem;">Reward</div>
        <div class="big-metric {reward_color}">{reward:.1f}</div>
        <div style="color: #666; font-size: 0.8rem;">current</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Update Charts using Plotly (only update every N steps for better performance)
    if len(st.session_state.simulation_data['time_steps']) > 1:
        
        # CO2 Chart
        co2_fig = create_co2_chart(
            st.session_state.simulation_data['time_steps'],
            st.session_state.simulation_data['co2_history'],
            env.co2_setpoint,
            env.co2_outdoor
        )
        co2_chart_placeholder.plotly_chart(co2_fig, use_container_width=True, key=f"co2_{st.session_state.current_step}")
        
        # Fan Speed Chart
        fan_fig = create_fan_chart(
            st.session_state.simulation_data['time_steps'],
            st.session_state.simulation_data['fan_speed_history']
        )
        fan_chart_placeholder.plotly_chart(fan_fig, use_container_width=True, key=f"fan_{st.session_state.current_step}")
        
        # Combined Chart
        combined_fig = create_combined_chart(
            st.session_state.simulation_data['time_steps'],
            st.session_state.simulation_data['energy_history'],
            st.session_state.simulation_data['occupancy_history'],
            env.max_occupancy
        )
        energy_chart_placeholder.plotly_chart(combined_fig, use_container_width=True, key=f"energy_{st.session_state.current_step}")
    
    # Update Detail Info
    step_placeholder.metric("Step", f"{st.session_state.current_step}/{env.max_steps}")
    time_placeholder.metric("Time", f"{current_time:.2f} hours")
    violations_placeholder.metric("IAQ Violations", info['iaq_violations'])
    compliance_placeholder.metric("Compliance Rate", f"{info['iaq_compliance_rate']*100:.1f}%")
    
    # Check if episode ended
    if terminated or truncated:
        st.session_state.simulation_running = False
        st.success(f"""
        ✅ **Episode Complete!**
        - Total Reward: {info['total_reward']:.2f}
        - IAQ Compliance: {info['iaq_compliance_rate']*100:.1f}%
        - Energy Consumed: {info['energy_consumed']:.2f} Wh
        - IAQ Violations: {info['iaq_violations']}
        """)
        st.balloons()
    
    # Control simulation speed
    time.sleep(0.05 / simulation_speed)
    
    # Rerun to continue simulation
    st.rerun()

else:
    # Simulation paused - show current state
    if st.session_state.observation is not None:
        st.info("⏸️ Simulation paused. Click 'Start' to continue or 'Reset' to restart.")
        
        # Show current metrics even when paused
        observation = st.session_state.observation
        env = st.session_state.env
        
        co2_level = observation[0]
        co2_status = "status-good" if co2_level < env.co2_setpoint else "status-danger"
        
        co2_placeholder.markdown(f"""
        <div class="metric-box">
            <div style="color: #666; font-size: 0.9rem;">CO₂ Level</div>
            <div class="big-metric {co2_status}">{co2_level:.0f}</div>
            <div style="color: #666; font-size: 0.8rem;">ppm</div>
        </div>
        """, unsafe_allow_html=True)
        
        fan_placeholder.markdown(f"""
        <div class="metric-box">
            <div style="color: #666; font-size: 0.9rem;">Fan Speed</div>
            <div class="big-metric">{observation[1]*100:.0f}</div>
            <div style="color: #666; font-size: 0.8rem;">%</div>
        </div>
        """, unsafe_allow_html=True)
        
        occupancy_placeholder.markdown(f"""
        <div class="metric-box">
            <div style="color: #666; font-size: 0.9rem;">Occupancy</div>
            <div class="big-metric">{int(observation[3])}</div>
            <div style="color: #666; font-size: 0.8rem;">people</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show last known charts
        if len(st.session_state.simulation_data['time_steps']) > 1:
            co2_fig = create_co2_chart(
                st.session_state.simulation_data['time_steps'],
                st.session_state.simulation_data['co2_history'],
                env.co2_setpoint,
                env.co2_outdoor
            )
            co2_chart_placeholder.plotly_chart(co2_fig, use_container_width=True)
            
            fan_fig = create_fan_chart(
                st.session_state.simulation_data['time_steps'],
                st.session_state.simulation_data['fan_speed_history']
            )
            fan_chart_placeholder.plotly_chart(fan_fig, use_container_width=True)
            
            combined_fig = create_combined_chart(
                st.session_state.simulation_data['time_steps'],
                st.session_state.simulation_data['energy_history'],
                st.session_state.simulation_data['occupancy_history'],
                env.max_occupancy
            )
            energy_chart_placeholder.plotly_chart(combined_fig, use_container_width=True)
    else:
        st.info("▶️ Click 'Start' to begin the simulation")