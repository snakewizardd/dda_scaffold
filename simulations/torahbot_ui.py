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
    @import url('https://fonts.googleapis.com/css2?family=Frank+Ruhl+Libre:wght@400;700&display=swap');
    
    .hebrew { 
        font-family: 'Frank Ruhl Libre', serif;
        font-size: 1.5rem;
        direction: rtl;
        text-align: right;
        color: #ffd700;
        background: rgba(255,215,0,0.1);
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        line-height: 2;
    }
    .stApp { background: #1a1a2e; color: #e0e0e6; }
    .stChatMessage { border-bottom: 1px solid rgba(255,255,255,0.05); }
</style>
""", unsafe_allow_html=True)

# Init
if "bot" not in st.session_state:
    st.session_state.bot = TorahBotSim()
    st.session_state.chat = [{"role": "assistant", "content": "שלום! Paste Hebrew text below and give a translation attempt."}]
    st.session_state.ready = False

bot = st.session_state.bot

# Sidebar
with st.sidebar:
    st.markdown("### 📖 Current Passage")
    if bot.study.hebrew_text:
        st.markdown(f'<div class="hebrew">{bot.study.hebrew_text}</div>', unsafe_allow_html=True)
    else:
        st.info("No Hebrew set")
    
    st.markdown(f"**Band:** {bot.agent.band} | **ρ:** {bot.agent.rho:.3f}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset / Next Pasuk"):
            bot.study.reset_passage()
            st.session_state.chat = [{"role": "assistant", "content": "Ready for the next passage! Paste Hebrew text below."}]
            st.rerun()
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat = []
            st.rerun()

# Main
st.markdown("## 📜 TorahBot")

# Hebrew input
with st.expander("Set Hebrew Passage", expanded=not bool(bot.study.hebrew_text)):
    heb = st.text_area("Hebrew", placeholder="אָוֶן יַחְשֹׁב עַל־מִשְׁכָּבוֹ", height=80)
    if st.button("Set"):
        if heb.strip():
            bot.study.hebrew_text = heb.strip()
            bot.study.last_attempt_text = None
            bot.study.attempt_present = False
            st.success("Set!")
            st.rerun()

# Chat
for msg in st.session_state.chat:
    with st.chat_message(msg["role"], avatar="📜" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Translation attempt or question..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar="📜"):
        ph = st.empty()
        ph.markdown("*thinking...*")
        
        async def run():
            if not st.session_state.ready:
                await bot.initialize_embeddings()
                st.session_state.ready = True
            return await bot.process_turn(prompt)
        
        resp = asyncio.run(run())
        ph.markdown(resp)
        st.session_state.chat.append({"role": "assistant", "content": resp})
        st.rerun()
