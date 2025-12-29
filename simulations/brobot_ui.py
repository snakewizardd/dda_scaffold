import streamlit as st
import asyncio
import time
import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path
from true_brobot import TrueBrobot, CONFIG, D1_PARAMS, C, BROBOT_CONFIG

# Ensure page config is first
st.set_page_config(
    page_title="TRUE BROBOT",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium SOTA Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --bg-color: #0f0f13;
        --card-bg: rgba(255, 255, 255, 0.05);
        --accent-primary: #7289da;
        --accent-success: #a6e3a1;
        --accent-warning: #f9e2af;
        --accent-danger: #eba0ac;
        --text-main: #e0e0e6;
        --text-dim: #9494b8;
    }

    .stApp {
        background: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #0f0f13 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-main);
    }

    /* Hide Streamlit Header/Footer */
    header, footer { visibility: hidden !important; height: 0 !important; }
    #MainMenu { visibility: hidden !important; }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.2); }

    /* Sidebar Glassmorphism */
    div[data-testid="stSidebar"] {
        background-color: rgba(15, 15, 19, 0.7) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Card Styling */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }

    /* Chat Styling */
    .stChatMessage {
        background-color: transparent !important;
        padding: 1rem 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    /* Animation for chat messages */
    .stChatMessage {
        animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Input Area */
    .stChatInput {
        padding-bottom: 2rem !important;
    }
    .stChatInput input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 12px !important;
    }

    /* Typography */
    h1, h2, h3 {
        font-weight: 800;
        letter-spacing: -0.02em;
        color: white !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace;
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: #fca7ea !important;
        padding: 0.2em 0.4em !important;
        border-radius: 4px !important;
    }

    /* Progress Bars */
    .stProgress > div > div > div > div {
        background-color: var(--accent-primary) !important;
    }

    /* Custom Metric Cards */
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        margin: 0;
    }
    .metric-label {
        color: var(--text-dim);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "brobot" not in st.session_state:
    st.session_state.brobot = TrueBrobot()
    st.session_state.chat_history = []
    # Add initial greeting if empty
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": "yo what's good! I'm here. what kinda support you looking for today? hype? clarity? planning? or just being heard?"
    })
    st.session_state.metrics_history = []

# PERSISTENCE FUNCTIONS
SESSION_FILE = Path("data/brobot/latest_session.pkl")

def save_session_state():
    """Save the full brobot instance and chat history to a pickle file."""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    state_dump = {
        "brobot": st.session_state.brobot,
        "chat_history": st.session_state.chat_history,
        "metrics_history": st.session_state.metrics_history,
        "timestamp": datetime.now().isoformat()
    }
    with open(SESSION_FILE, "wb") as f:
        pickle.dump(state_dump, f)
    return True

def load_session_state():
    """Load session from pickle file."""
    if SESSION_FILE.exists():
        with open(SESSION_FILE, "rb") as f:
            state_dump = pickle.load(f)
            st.session_state.brobot = state_dump["brobot"]
            # HOT PATCH: Apply latest config (Smart Bro Upgrade) to restored instance
            st.session_state.brobot.config = BROBOT_CONFIG
            
            st.session_state.chat_history = state_dump["chat_history"]
            st.session_state.metrics_history = state_dump["metrics_history"]
        return True
    return False

def get_band_color(band):
    colors = {
        "PRESENT": "#a6e3a1",   # Green
        "AWARE": "#f9e2af",     # Yellow
        "WATCHFUL": "#fab387",  # Orange
        "CONTRACTED": "#eba0ac",# Red/Pink
        "FROZEN": "#89b4fa"     # Blue
    }
    return colors.get(band, "#cdd6f4")

# Sidebar - Cognitive Dashboard
with st.sidebar:
    st.markdown("<h1>🧠 DASHBOARD</h1>", unsafe_allow_html=True)
    
    agent = st.session_state.brobot.agent
    current_band = agent.band
    band_color = get_band_color(current_band)
    
    # Current State Indicators
    st.markdown(f"""
    <div class="glass-card">
        <div class="status-badge" style="background: {band_color}22; color: {band_color}; border: 1px solid {band_color}44;">
            ● {current_band}
        </div>
        <div class="metric-label">Rigidity (ρ)</div>
        <div class="metric-value" style="color: {band_color}">{agent.rho:.3f}</div>
        <div style="margin-top: 10px; font-size: 0.7em; color: var(--text-dim);">
            Active Band: {current_band}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="padding: 15px;">
            <div class="metric-label">Arousal</div>
            <div class="metric-value" style="font-size: 1.2rem;">{agent.arousal:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="padding: 15px;">
            <div class="metric-label">Trust</div>
            <div class="metric-value" style="font-size: 1.2rem; color: var(--accent-success);">{agent.user_trust*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🛠️ Controls", unsafe_allow_html=True)
    
    # Physics State
    st.progress(min(1.0, agent.rho), text=f"Rigidity: {agent.rho:.2f}")
    st.progress(min(1.0, agent.arousal), text=f"Arousal: {agent.arousal:.2f}")
    st.progress(min(1.0, agent.user_trust), text=f"Bro-Trust: {agent.user_trust*100:.0f}%")
    
    if st.session_state.brobot.see_me_mode:
        st.warning("👁️ SEE ME MODE ACTIVE", icon="👁️")
        
    st.divider()
    
    # Report Bug / Save Session
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("💾 Save"):
            if save_session_state():
                st.success("Saved!")
    with col_s2:
         if st.button("📂 Load"):
             if load_session_state():
                 st.success("Loaded!")
                 st.rerun()
    
    # Live Metrics Graph
    if len(st.session_state.metrics_history) > 2:
        df = pd.DataFrame(st.session_state.metrics_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['rho'], name='Rigidity', line=dict(color='#fab387', width=2)))
        fig.add_trace(go.Scatter(y=df['epsilon'], name='Surprise', line=dict(color='#89b4fa', width=1)))
        
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            font=dict(color='#cdd6f4', size=10),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=True, gridcolor='#313244', range=[0, 1.2])
        )
        st.plotly_chart(fig, use_container_width=True)

# Main Chat Area
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🤝 TRUE BROBOT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: var(--text-dim); margin-bottom: 2rem;'>Your Ride-or-Die Digital Companion</p>", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Say something..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Process response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("typing...")
        
        # Async run wrapper
        async def run_turn():
            response = await st.session_state.brobot.process_user_input(prompt)
            return response
        
        response = asyncio.run(run_turn())
        
        # Update UI
        message_placeholder.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Log metrics for graph
        last_log = st.session_state.brobot.session_log[-1]
        metrics = last_log["agent_metrics"]
        st.session_state.metrics_history.append({
            "rho": metrics["rho_after"],
            "epsilon": metrics.get("input_epsilon", 0.0), # Tracking input surprise specifically
            "band": metrics["band"]
        })
        
        # Autosave
        save_session_state()
        
        # Force rerun to update sidebar
        st.rerun()
