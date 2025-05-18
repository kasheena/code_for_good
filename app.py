import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Set page config FIRST
st.set_page_config(page_title="LevelHead: Minimal Mental Health", layout="centered", page_icon="⬛️")

# Custom CSS for minimal black & white look
st.markdown("""
    <style>
    html, body, .stApp {
        background: #fff !important;
        color: #111 !important;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    }
    .stButton > button {
        background: #111 !important;
        color: #fff !important;
        border-radius: 4px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stProgress > div > div {
        background: #111 !important;
    }
    hr {border: 1px solid #eee;}
    </style>
""", unsafe_allow_html=True)

# --- Problem Statement ---
st.title("LevelHead ⬛️")
st.caption("Minimal. Gameful. Mental Health.")

with st.expander("📝 Problem & Solution (Pitch-Ready)"):
    st.markdown("""
    **Problem:**  
    Mental health issues often go undetected until they become critical. Most digital tools are either overwhelming or too clinical for daily engagement.

    **Solution:**  
    **LevelHead** makes self-care simple, stigma-free, and engaging. Track your mood, complete daily micro-challenges, and earn streaks-all in a minimal, black-and-white interface.
    """)

st.markdown("---")

# --- Gamified Daily Check-in ---
st.header("🎯 Daily Check-in")
today = datetime.now().date()
mood = st.slider("How do you feel today?", 0, 10, 5, format="%d")
if 'checkins' not in st.session_state:
    st.session_state['checkins'] = {}
if st.button("Submit Check-in"):
    st.session_state['checkins'][str(today)] = mood
    st.success("Check-in recorded!")

# --- Streaks & Progress ---
checkins = st.session_state['checkins']
dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in checkins.keys()]
dates.sort()
streak = 1
for i in range(len(dates)-1, 0, -1):
    if (dates[i] - dates[i-1]).days == 1:
        streak += 1
    else:
        break

st.markdown(f"**Current Streak:** `{streak}` days")
if streak and streak % 7 == 0:
    st.balloons()
    st.success("🏆 7-Day Streak! Badge Unlocked.")

# --- Micro-Challenge ---
st.header("🕹️ Micro-Challenge")
challenges = [
    "Take a 5-minute mindful break",
    "Send a positive message to a friend",
    "Go for a short walk",
    "Write down one thing you're grateful for",
    "Drink a glass of water mindfully"
]
challenge_idx = (today.day + streak) % len(challenges)
st.markdown(f"**Today's Challenge:** {challenges[challenge_idx]}")

if st.button("Mark Challenge as Done"):
    st.success("Challenge completed! +1 XP")

# --- Mood History (Minimal Chart) ---
if checkins:
    st.header("📈 Mood History")
    df = pd.DataFrame(list(checkins.items()), columns=['Date', 'Mood'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    st.line_chart(df.set_index('Date')['Mood'])
else:
    st.info("No check-ins yet. Start today to build your streak!")

st.markdown("---")

# --- Footer ---
st.caption("LevelHead ⬛️ – Less chaos. More clarity. Built for hackathons and everyday life.")

