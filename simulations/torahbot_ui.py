"""
TORAHBOT UI — Minimal Streamlit wrapper for RTL Hebrew display
Run: streamlit run simulations/torahbot_ui.py
"""

import streamlit as st
import asyncio
from torah_study_bot import TorahBotSim

st.set_page_config(page_title="TorahBot", page_icon="📜", layout="wide")

# RTL Hebrew styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;700&family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .hebrew-large { 
        font-family: 'Frank Ruhl Libre', serif;
        font-size: 2rem;
        direction: rtl;
        text-align: right;
        color: #ffd700;
        background: rgba(255,215,0,0.05);
        padding: 20px;
        border-right: 4px solid #ffd700;
        border-radius: 4px;
        margin: 15px 0;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        white-space: pre-wrap;
    }
    
    .hebrew-inline { 
        font-family: 'Frank Ruhl Libre', serif;
        direction: rtl;
        text-align: right;
        display: inline-block;
        color: #ffd700;
    }

    .candidate-box {
        background: rgba(255,255,255,0.03);
        padding: 15px;
        border-radius: 10px;
        border-left: 3px solid #444;
        margin-bottom: 15px;
    }
    
    .selected-candidate {
        border-left: 5px solid #00ff88;
        background: rgba(0,255,136,0.05);
    }

    .stApp { background: #0e1117; color: #e0e0e6; }
</style>
""", unsafe_allow_html=True)

# Init
if "bot" not in st.session_state:
    st.session_state.bot = TorahBotSim()
    st.session_state.chat = [{"role": "assistant", "content": "Welcome to **TorahBot**. I am your chevrutah for Biblical Hebrew study. Paste a verse below to begin."}]
    st.session_state.ready = False

bot = st.session_state.bot

# Sidebar
with st.sidebar:
    st.title("📜 TorahBot Control")
    st.markdown("---")
    
    st.markdown("### 🏛️ State")
    st.markdown(f"**Current Band:** `{bot.agent.band}`")
    st.progress(max(0.0, min(1.0, bot.agent.rho)), text=f"Rigidity (ρ): {bot.agent.rho:.3f}")
    st.progress(max(0.0, min(1.0, bot.agent.arousal)), text=f"Arousal: {bot.agent.arousal:.3f}")
    
    st.markdown("---")
    st.markdown("### 📖 Current Text")
    if bot.study.hebrew_text:
        st.markdown(f'<div class="hebrew-large">{bot.study.hebrew_text}</div>', unsafe_allow_html=True)
    else:
        st.info("No passage set yet.")
    
    if st.button("Reset Session", use_container_width=True):
        bot.study.reset_passage()
        st.session_state.chat = [{"role": "assistant", "content": "Session reset. Enter a new passage below."}]
        st.rerun()

# Main
st.title("DDA-X Hebrew Exegesis")

# Hebrew input
heb_col1, heb_col2 = st.columns([4, 1])
with heb_col1:
    heb_input = st.text_input("Hebrew Pasuk", placeholder="e.g. בראשית ברא אלהים...", value=bot.study.hebrew_text or "")
with heb_col2:
    if st.button("Set Passage", use_container_width=True):
        if heb_input.strip():
            bot.study.hebrew_text = heb_input.strip()
            bot.study.last_attempt_text = None
            bot.study.attempt_present = False
            st.rerun()

st.divider()

# Chat display
for i, msg in enumerate(st.session_state.chat):
    with st.chat_message(msg["role"], avatar="📜" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])
        
        # Render Dashboard if present in this message
        metrics = msg.get("dda_metrics")
        if metrics and "candidates_debug" in metrics:
            with st.expander(f"🧠 DDA-X CORRIDOR DASHBOARD", expanded=(i == len(st.session_state.chat)-1)):
                cands = metrics["candidates_debug"]
                
                # Dashboard Summary
                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.metric("Selected J-Score", f"{metrics.get('J_final', 0):.3f}")
                s_col2.metric("Efficiency (K)", f"{len(cands)}")
                grad = "🟢" if not metrics.get("corridor_failed") else "🔴"
                s_col3.metric("Corridor Status", f"{grad} {'PASS' if not metrics.get('corridor_failed') else 'FALLBACK'}")
                
                st.divider()
                
                # Per-candidate detail in Tabs
                tab_labels = []
                for idx, c in enumerate(cands):
                    status_icon = "✅" if c.get("is_chosen") else "❌"
                    j_val = c.get("J_final", 0)
                    tab_labels.append(f"{status_icon} Option {idx+1} (J={j_val:.2f})")
                
                tabs = st.tabs(tab_labels)
                
                for idx, tab in enumerate(tabs):
                    with tab:
                        c = cands[idx]
                        m = c.get("metrics", {})
                        if c.get("is_chosen"):
                            st.success("**OPTIMAL CANDIDATE SELECTED**")
                        else:
                            st.error("**REJECTED BY CORRIDOR**")
                        
                        st.markdown(f"**J:** `{c.get('J_final', 0):.4f}` | **E:** `{m.get('E', 0):.4f}` | **N:** `{m.get('novelty', 0):.4f}`")
                        
                        # Render candidate text. If it contains Hebrew, wrap in RTL div.
                        # Simple heuristic: if it has Hebrew characters
                        cand_text = c.get("text", "")
                        import re
                        if re.search(r"[\u0590-\u05FF]", cand_text):
                            st.markdown(f'<div class="hebrew-large" style="font-size:1.1rem; padding:10px;">{cand_text}</div>', unsafe_allow_html=True)
                        else:
                            st.info("**Draft Content:**")
                            st.code(cand_text, language=None)

# User Input
if prompt := st.chat_input("Enter your translation attempt or a question about the text..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    st.rerun()

# Check if last message was user to process
if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
    last_user_msg = st.session_state.chat[-1]["content"]
    
    with st.chat_message("assistant", avatar="📜"):
        status_ph = st.empty()
        status_ph.info("🔄 **DDA-X System Initializing...** Generating K-candidates and evaluating corridor.")
        
        async def run_sim():
            if not st.session_state.ready:
                await bot.initialize_embeddings()
                st.session_state.ready = True
            return await bot.process_turn(last_user_msg)
        
        resp = asyncio.run(run_sim())
        status_ph.empty()
        
        if resp.get("type") in ["chat", "command"]:
            text = resp.get("text", "")
            metrics = resp.get("metrics")
            
            # Append WITH metrics so they persist
            st.session_state.chat.append({
                "role": "assistant", 
                "content": text,
                "dda_metrics": metrics
            })
            st.rerun()
            
        elif resp.get("type") == "quit":
            st.markdown("Session ending. Logs saved.")
            st.stop()
