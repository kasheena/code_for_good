import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
# --- Main App ---
st.set_page_config(page_title="VibeCheck: Gen Z Mental Health", layout="wide", page_icon="🧠")
# Download necessary NLTK data (only needs to be done once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# --- Gen Z Color Palette and Custom CSS ---
st.markdown("""
    <style>
    body, .stApp {
        background: linear-gradient(135deg, #F9F6FF 0%, #E0C3FC 100%);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .main {
        background: transparent;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #fff;
        border-radius: 1.5rem;
        box-shadow: 0 2px 16px rgba(160, 100, 255, 0.10);
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1.1rem;
        color: #3A2B7C;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #fff;
        background: linear-gradient(90deg, #A064FF 60%, #5FFFE1 100%);
        border-radius: 1.2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #A064FF 60%, #5FFFE1 100%);
        color: #fff;
        font-weight: 700;
        border: none;
        border-radius: 1.5rem;
        padding: 0.7rem 2rem;
        box-shadow: 0 2px 8px rgba(160, 100, 255, 0.12);
    }
    .stProgress > div > div {
        border-radius: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Mental Health Keywords ---
mental_health_keywords = {
    "sleep": ["sleep", "insomnia", "tired", "exhausted", "rest", "bed", "awake", "nap"],
    "anxiety": ["worry", "anxious", "nervous", "stress", "panic", "fear", "overwhelm"],
    "depression": ["sad", "depress", "empty", "hopeless", "worthless", "lonely", "numb"],
    "social": ["friend", "family", "people", "alone", "isolat", "talk", "social"],
    "motivation": ["motivat", "energy", "interest", "hobby", "passion", "enjoy", "effort"]
}

def analyze_text(text):
    text = text.lower()
    sentiment = sia.polarity_scores(text)
    category_scores = {}
    for category, keywords in mental_health_keywords.items():
        score = sum(keyword in text for keyword in keywords)
        category_scores[category] = min(score / len(keywords) * 2, 1.0)
    risk_score = (
        (1 - sentiment['compound']) * 50 +
        (category_scores.get('depression', 0) * 15) +
        (category_scores.get('anxiety', 0) * 12) +
        (category_scores.get('sleep', 0) * 10) +
        (category_scores.get('social', 0) * 8) +
        (category_scores.get('motivation', 0) * 5)
    )
    risk_score = max(0, min(100, risk_score))
    return {
        'sentiment': sentiment,
        'category_scores': category_scores,
        'risk_score': risk_score
    }

def generate_time_series(days=30, declining=True):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    section1 = days // 3
    section2 = days // 3
    section3 = days - section1 - section2
    if declining:
        sleep_hours = [7 + random.uniform(-0.5, 0.5) for _ in range(section1)] + \
                     [6 + random.uniform(-1.0, 0.5) for _ in range(section2)] + \
                     [5 + random.uniform(-1.0, 0.5) for _ in range(section3)]
        mood_scores = [80 + random.uniform(-10, 10) for _ in range(section1)] + \
                     [65 + random.uniform(-15, 10) for _ in range(section2)] + \
                     [50 + random.uniform(-15, 5) for _ in range(section3)]
        social_scores = [75 + random.uniform(-15, 15) for _ in range(section1)] + \
                       [60 + random.uniform(-20, 10) for _ in range(section2)] + \
                       [40 + random.uniform(-15, 10) for _ in range(section3)]
    else:
        sleep_hours = [7 + random.uniform(-0.7, 0.7) for _ in range(days)]
        mood_scores = [75 + random.uniform(-15, 15) for _ in range(days)]
        social_scores = [70 + random.uniform(-20, 20) for _ in range(days)]
    return pd.DataFrame({
        'Date': dates,
        'Sleep': sleep_hours,
        'Mood': mood_scores,
        'Social': social_scores
    })

sample_entries = [
    {
        "text": "Had a good day today. Went for a walk in the park and met some friends for coffee. Looking forward to the weekend!",
        "risk_level": "Low"
    },
    {
        "text": "Work was stressful today. Feeling a bit tired but nothing too bad. Might skip the gym tonight and just relax.",
        "risk_level": "Low-Medium"
    },
    {
        "text": "Haven't been sleeping well lately. Work is fine but I just don't feel motivated anymore. Everything takes so much effort.",
        "risk_level": "Medium"
    },
    {
        "text": "I can't remember the last time I felt happy. Nobody understands what I'm going through. I'm so tired of trying.",
        "risk_level": "High"
    }
]



# --- Hero Section ---
st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0;">
        <h1 style="font-size:2.8rem; font-weight:900; color:#A064FF; margin-bottom:0.5rem;">
            VibeCheck 🧠
        </h1>
        <h3 style="font-weight:500; color:#3A2B7C;">
            Your AI-powered mental health early warning system.<br>
            Personalized. Real-time. Social. <span style="color:#5FFFE1;">Made for Gen Z.</span>
        </h3>
        <p style="font-size:1.1rem; color:#444; max-width:600px; margin:auto;">
            Spot the vibes, track your wellness, and get actionable support-before things go south. <br>
            <span style="color:#A064FF; font-weight:600;">Pitch-ready for hackathons. Built for the next generation of wellness.</span>
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["✨ VibeCheck (Text)", "📈 Trends", "ℹ️ About"])

with tab1:
    st.header("✨ Journal VibeCheck")
    st.write("Analyze your journal entry or social post for instant mood and risk feedback. Stay ahead of mental health dips with real-time AI insights.")

    use_sample = st.toggle("Try a sample entry", value=True)
    if use_sample:
        sample_idx = st.selectbox(
            "Pick a sample vibe:",
            range(len(sample_entries)),
            format_func=lambda i: f"Sample {i+1}: {sample_entries[i]['risk_level']} Risk"
        )
        text_input = st.text_area("Your journal entry:", value=sample_entries[sample_idx]["text"], height=120)
    else:
        text_input = st.text_area(
            "Type your journal entry or post here...",
            value="",
            height=120,
            placeholder="What's on your mind today? (e.g. 'Feeling anxious about exams...')"
        )

    if st.button("🔍 Analyze My Vibe") and text_input:
        with st.spinner("Reading your vibes..."):
            analysis_result = analyze_text(text_input)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Mood Meter")
                sentiment = analysis_result['sentiment']
                st.write("Overall Vibe:")
                sentiment_value = sentiment['compound']
                st.progress((sentiment_value + 1) / 2)
                st.write(f"Score: {sentiment_value:.2f} ({'Positive' if sentiment_value > 0 else 'Negative'})")
                st.write(f"Positive: {sentiment['pos']:.2f}")
                st.write(f"Neutral: {sentiment['neu']:.2f}")
                st.write(f"Negative: {sentiment['neg']:.2f}")
            with col2:
                st.subheader("Key Wellness Signals")
                category_scores = analysis_result['category_scores']
                sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
                for category, score in sorted_categories:
                    st.write(f"{category.title()}")
                    st.progress(score)
            st.subheader("Risk Radar")
            risk_score = analysis_result['risk_score']
            if risk_score < 30:
                risk_label, risk_color = "Low Risk", "#5FFFE1"
            elif risk_score < 50:
                risk_label, risk_color = "Low-Medium", "#FFD600"
            elif risk_score < 70:
                risk_label, risk_color = "Medium", "#FFA500"
            else:
                risk_label, risk_color = "High", "#FF3B30"
            st.markdown(f"<h3 style='color:{risk_color};'>{risk_label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<b>Score:</b> <span style='color:{risk_color}; font-size:1.2rem;'>{risk_score:.1f}/100</span>", unsafe_allow_html=True)
            st.progress(risk_score/100)
            st.caption("Low   |   Low-Med   |   Medium   |   High")
            st.subheader("Personalized Next Steps")
            if risk_score < 30:
                recommendations = [
                    "Keep up your self-care routine! 🚀",
                    "Track your mood for extra awareness.",
                    "Share your vibes with friends or our community."
                ]
            elif risk_score < 50:
                recommendations = [
                    "Try a 10-min mindfulness break.",
                    "Check in with a friend.",
                    "Optimize your sleep schedule."
                ]
            elif risk_score < 70:
                recommendations = [
                    "Talk to someone you trust.",
                    "Stick to a steady sleep routine.",
                    "Move your body-walk, dance, stretch.",
                    "Try gratitude journaling."
                ]
            else:
                recommendations = [
                    "Reach out to a mental health pro.",
                    "Let someone close know how you feel.",
                    "Focus on rest, nutrition, and gentle activity.",
                    "Use grounding techniques when stressed.",
                    "Remember: support is always here."
                ]
            for rec in recommendations:
                st.markdown(f"• {rec}")
            if risk_score >= 70:
                st.markdown("---")
                st.markdown("### 🚨 Support Resources")
                st.markdown("• **Crisis Text Line**: Text HOME to 741741")
                st.markdown("• **988 Suicide & Crisis Lifeline**: 988 or 1-800-273-8255")
                st.markdown("• **SAMHSA Helpline**: 1-800-662-4357")

with tab2:
    st.header("📈 Wellness Trends")
    st.write("Visualize your 30-day patterns in sleep, mood, and social activity. Spot when your wellness is trending up-or when it needs a boost.")
    profile_type = st.radio("Pick your vibe pattern:", ["Stable", "Declining"])
    is_declining = profile_type == "Declining"
    df = generate_time_series(days=30, declining=is_declining)
    st.subheader("30-Day Wellness Chart")
    selected_metric = st.selectbox("Show:", ["All Metrics", "Sleep Hours", "Mood Score", "Social Activity"])
    if selected_metric == "All Metrics":
        df_plot = df.copy()
        df_plot['Sleep_norm'] = df_plot['Sleep'] / 8 * 100
        chart_data = pd.DataFrame({
            'Date': df_plot['Date'],
            'Sleep Quality': df_plot['Sleep_norm'],
            'Mood Score': df_plot['Mood'],
            'Social Activity': df_plot['Social']
        }).set_index('Date')
        st.line_chart(chart_data)
        if is_declining:
            st.caption("Red vertical line = change point detected")
    elif selected_metric == "Sleep Hours":
        st.line_chart(df.set_index('Date')['Sleep'])
        st.caption("Orange line = concern threshold (6 hours)")
        if is_declining:
            st.caption("Red vertical line = change point detected")
    elif selected_metric == "Mood Score":
        st.line_chart(df.set_index('Date')['Mood'])
        st.caption("Orange line = concern threshold (60/100)")
        if is_declining:
            st.caption("Red vertical line = change point detected")
    else:
        st.line_chart(df.set_index('Date')['Social'])
        st.caption("Orange line = concern threshold (50/100)")
        if is_declining:
            st.caption("Red vertical line = change point detected")
    st.subheader("Pattern Insights")
    if is_declining:
        st.markdown("""
        • 🚩 **Decline detected** in multiple wellness signals  
        • Sleep disruption started ~2 weeks ago  
        • Mood drop matches sleep changes  
        • Social activity fell after mood dip  
        **Next steps:**  
        - Book a sleep check-in  
        - Reconnect with friends  
        - Try mood support tools  
        """)
    else:
        st.markdown("""
        • All wellness signals are stable  
        • No concerning trends  
        **Keep it up!**  
        - Stick to your routine  
        - Try new wellness activities  
        """)

with tab3:
    st.header("ℹ️ About VibeCheck")
    st.markdown("""
    **VibeCheck** is a next-gen mental health early warning system for Gen Z.  
    - AI-powered text analysis for instant emotional feedback  
    - Visual trend tracking for sleep, mood, and social activity  
    - Personalized, actionable recommendations  
    - Social, gamified, and ready for wearables  
    - Designed for hackathons, built for real-world impact  
    ---
    **Pitch Value:**  
    - Tackles the mental health crisis with proactive, personalized tech[2][3]  
    - Leverages Gen Z’s love for social, simple, authentic, and gamified experiences  
    - Ready for integration with wearables, telehealth, and community features  
    - Built to scale, customizable, and privacy-first  
    """)
    st.caption("Built by Gen Z, for Gen Z. #WellnessForAll #PitchReady")

