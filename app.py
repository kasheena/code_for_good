import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="BlackBox: Mood Tracing", layout="centered", page_icon="⬛️")

# Minimal black & white CSS
st.markdown("""
    <style>
    html, body, .stApp { background: #fff !important; color: #111 !important; font-family: 'Inter', Arial, sans-serif; }
    .stButton > button { background: #111 !important; color: #fff !important; border-radius: 4px; border: none; font-weight: 600; }
    .stProgress > div > div { background: #111 !important; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("⬛️ BlackBox")
st.caption("Passive mood tracing. No words needed.")

with st.expander("What is BlackBox?"):
    st.markdown("""
    - **No journaling, no mood sliders, no chaos.**
    - Detects mental health drift using only behavioral signals (typing rhythm, app switching, device usage).
    - **No content ever leaves your device.**
    - Shows a single, private “BlackBox Signal” waveform.
    - If your signal drifts, it’s time to check in with yourself.
    """)

st.markdown("---")

# Simulate a behavioral signal drift (for demo)
np.random.seed(42)
N = 60
signal = np.cumsum(np.random.randn(N) * 0.3)
# Insert a "drift" event
drift_point = 40
signal[drift_point:] += np.linspace(0, -8, N-drift_point)

df = pd.DataFrame({'Time': range(N), 'BlackBox Signal': signal})

if st.button("Check My BlackBox"):
    st.line_chart(df.set_index('Time'), use_container_width=True)
    drift_value = signal[-1] - signal[0]
    if drift_value < -5:
        st.markdown("### ⚠️ Drift Detected")
        st.write("Your recent digital patterns suggest a significant change in your mood or energy.")
        st.write("This is a private signal. If you want, take a moment to reflect or reach out to someone you trust.")
    else:
        st.markdown("### ✔️ All Clear")
        st.write("Your BlackBox signal is stable. Keep going.")

st.markdown("---")
st.caption("BlackBox ⬛️ – For when you can't put it into words.")

