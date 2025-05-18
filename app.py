import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Download necessary NLTK data (only needs to be done once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load a simple mental health keyword dictionary
mental_health_keywords = {
    "sleep": ["sleep", "insomnia", "tired", "exhausted", "rest", "bed", "awake", "nap"],
    "anxiety": ["worry", "anxious", "nervous", "stress", "panic", "fear", "overwhelm"],
    "depression": ["sad", "depress", "empty", "hopeless", "worthless", "lonely", "numb"],
    "social": ["friend", "family", "people", "alone", "isolat", "talk", "social"],
    "motivation": ["motivat", "energy", "interest", "hobby", "passion", "enjoy", "effort"]
}

# Simple function to analyze text without external APIs
def analyze_text(text):
    text = text.lower()

    # Sentiment analysis
    sentiment = sia.polarity_scores(text)

    # Keyword matching
    category_scores = {}
    for category, keywords in mental_health_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        # Normalize score
        category_scores[category] = min(score / len(keywords) * 2, 1.0)

    # Calculate risk score (simplified algorithm)
    risk_score = (
        (1 - sentiment['compound']) * 50 +  # Convert sentiment to 0-50 scale
        (category_scores.get('depression', 0) * 15) +
        (category_scores.get('anxiety', 0) * 12) +
        (category_scores.get('sleep', 0) * 10) +
        (category_scores.get('social', 0) * 8) +
        (category_scores.get('motivation', 0) * 5)
    )

    # Ensure risk_score is in 0-100 range
    risk_score = max(0, min(100, risk_score))

    return {
        'sentiment': sentiment,
        'category_scores': category_scores,
        'risk_score': risk_score
    }

# Generate synthetic time series data
def generate_time_series(days=30, declining=True):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()

    # Calculate lengths for each section to ensure equal array lengths
    section1 = days // 3
    section2 = days // 3
    section3 = days - section1 - section2  # Ensure all sections add up to exactly 'days'

    if declining:
        # Generate declining mental health pattern
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
        # Generate stable pattern
        sleep_hours = [7 + random.uniform(-0.7, 0.7) for _ in range(days)]
        mood_scores = [75 + random.uniform(-15, 15) for _ in range(days)]
        social_scores = [70 + random.uniform(-20, 20) for _ in range(days)]

    # Verify all lists have the same length
    assert len(dates) == len(sleep_hours) == len(mood_scores) == len(social_scores), "Array length mismatch"

    return pd.DataFrame({
        'Date': dates,
        'Sleep': sleep_hours,
        'Mood': mood_scores,
        'Social': social_scores
    })

# Sample text entries with pre-classified risk levels
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

# Custom CSS for minimal black and white theme
custom_css = """
body {
    color: #333;
    background-color: #f8f8f8;
    font-family: sans-serif;
}
.st-header {
    background-color: #f0f0f0;
    padding: 1rem 0;
    border-bottom: 1px solid #eee;
}
h1, h2, h3, h4, h5, h6 {
    color: #111;
}
.st-tabs>ul>li>a {
    color: #555;
    background-color: #eee;
    border: 1px solid #ddd;
    border-bottom: none;
}
.st-tabs>ul>li[aria-selected="true"]>a {
    color: #111;
    background-color: #fff;
}
.st-button>button {
    color: #111;
    background-color: #eee;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    padding: 0.5rem 1rem;
    cursor: pointer;
}
.st-button>button:hover {
    background-color: #ddd;
}
.st-selectbox>div>div>div>div {
    color: #333;
    background-color: #fff;
    border: 1px solid #ccc;
    border-radius: 0.25rem;
    padding: 0.3rem;
}
.st-text-area>div>div>div>textarea {
    color: #333;
    background-color: #fff;
    border: 1px solid #ccc;
    border-radius: 0.25rem;
    padding: 0.5rem;
}
.st-checkbox>label>div {
    color: #333;
}
.st-progress>div>div>div>div {
    background-color: #888;
}
.streamlit-expander {
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    margin-bottom: 1rem;
}
.streamlit-expander-header {
    color: #333;
    background-color: #f0f0f0;
    padding: 0.5rem;
}
.streamlit-expander-content {
    padding: 0.5rem;
}
.st-markdown p, .st-markdown li {
    color: #333;
}
.st-markdown h3 {
    color: #222;
    border-bottom: 1px solid #eee;
    padding-bottom: 0.2rem;
    margin-bottom: 0.5rem;
}
"""

# Main application
def main():
    st.set_page_config(page_title="Mental Health Early Warning System", layout="wide")
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

    st.title("Mental Health Early Warning System")
    st.write("AI-powered early detection and intervention for mental health concerns")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Text Analysis", "Behavioral Trends", "About"])

    # Tab 1: Text Analysis
    with tab1:
        st.header("Journal Entry Analysis")

        # User can select a sample or enter their own text
        use_sample = st.checkbox("Use a sample entry", value=True)

        if use_sample:
            sample_idx = st.selectbox(
                "Select a sample journal entry:",
                range(len(sample_entries)),
                format_func=lambda i: f"Sample {i+1}: {sample_entries[i]['risk_level']} Risk"
            )
            text_input = sample_entries[sample_idx]["text"]
            text_input = st.text_area("Journal entry:", value=text_input, height=150)
        else:
            text_input = st.text_area(
                "Enter a journal entry or social media post to analyze:",
                value="",
                height=150,
                placeholder="Type your journal entry here..."
            )

        if st.button("Analyze Text") and text_input:
            with st.spinner("Analyzing text..."):
                # Analyze the text
                analysis_result = analyze_text(text_input)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sentiment Analysis")
                    sentiment = analysis_result['sentiment']

                    st.write("**Compound Sentiment Score:**")
                    sentiment_value = sentiment['compound']
                    st.progress((sentiment_value + 1) / 2)
                    st.write(f"Score: {sentiment_value:.2f} ({'Positive' if sentiment_value > 0 else 'Negative'})")

                    st.write(f"Positive: {sentiment['pos']:.2f}")
                    st.write(f"Neutral: {sentiment['neu']:.2f}")
                    st.write(f"Negative: {sentiment['neg']:.2f}")

                with col2:
                    st.subheader("Mental Health Indicators")

                    # Show category scores
                    category_scores = analysis_result['category_scores']
                    if category_scores:
                        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
                        for category, score in sorted_categories:
                            st.write(f"**{category.title()}**")
                            st.progress(score)
                            st.write(f"Score: {score:.2f}")

                # Risk assessment
                st.subheader("Risk Assessment")
                risk_score = analysis_result['risk_score']

                # Display risk level
                if risk_score < 30:
                    risk_label = "Low Risk"
                    risk_color = "green"
                elif risk_score < 50:
                    risk_label = "Low-Medium Risk"
                    risk_color = "yellow"
                elif risk_score < 70:
                    risk_label = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_label = "High Risk"
                    risk_color = "red"

                col1_risk, col2_risk = st.columns([1, 2])

                with col1_risk:
                    st.markdown(f"### {risk_label}")
                    st.markdown(f"Score: **{risk_score:.1f}/100**")

                with col2_risk:
                    st.progress(risk_score/100)
                    cols_labels = st.columns(4)
                    cols_labels[0].markdown("<p style='text-align: left;'>Low</p>", unsafe_allow_html=True)
                    cols_labels[1].markdown("<p style='text-align: center;'>Low-Med</p>", unsafe_allow_html=True)
                    cols_labels[2].markdown("<p style='text-align: center;'>Medium</p>", unsafe_allow_html=True)
                    cols_labels[3].markdown("<p style='text-align: right;'>High</p>", unsafe_allow_html=True)

                # Recommendations based on risk level
                st.subheader("Recommendations")

                if risk_score < 30:
                    recommendations = [
                        "Continue monitoring your mental well-being.",
                        "Maintain current self-care practices.",
                        "Consider using our mood tracking feature for ongoing awareness."
                    ]
                elif risk_score < 50:
                    recommendations = [
                        "Practice mindfulness for 10 minutes daily.",
                        "Ensure you're maintaining regular social connections.",
                        "Review and optimize your sleep hygiene."
                    ]
                elif risk_score < 70:
                    recommendations = [
                        "Consider speaking with a trusted friend about how you're feeling.",
                        "Implement a consistent sleep schedule.",
                        "Try daily physical activity, even short walks.",
                        "Practice gratitude journaling."
                    ]
                else:
                    recommendations = [
                        "**Reach out to a mental health professional.**",
                        "Talk to someone you trust about how you're feeling.",
                        "Focus on sleep, nutrition, and gentle physical activity.",
                        "Use grounding techniques when feeling overwhelmed.",
                        "**Remember that support is available.**"
                    ]

                for rec in recommendations:
                    st.markdown(f"• {rec}")

                if risk_score >= 70:
                    st.markdown("---")
                    st.subheader("Support Resources")
                    st.markdown("• **Crisis Text Line**: Text HOME to 741741")
                    st.markdown("• **National Suicide Prevention Lifeline**: 988 or 1-800-273-8255")
                    st.markdown("• **SAMHSA's National Helpline**: 1-800-662-4357")

    # Tab 2: Behavioral Trends
    with tab2:
        st.header("Behavioral Pattern Analysis")

        # Select profile type
        profile_type = st.radio(
            "Select profile type:",
            ["Stable Pattern", "Declining Pattern"]
        )

        # Generate appropriate data
        is_declining = profile_type == "Declining Pattern"
        df = generate_time_series(days=30, declining=is_declining)

        # Display time series data
        st.subheader("30-Day Behavioral Patterns")

        # Select metric to display
        selected_metric = st.selectbox(
            "Select metric to view:",
            ["All Metrics", "Sleep Hours", "Mood Score", "Social Activity"]
        )

        if selected_metric == "All Metrics":
            # Normalize data for comparison
            df_plot = df.copy()
            df_plot['Sleep_norm'] = df_plot['Sleep'] / 8 * 100  # Normalize to 0-100 scale

            # Use Streamlit's native line chart
            chart_data = pd.DataFrame({
                'Date': df_plot['Date'],
                'Sleep Quality': df_plot['Sleep_norm'],
                'Mood Score': df_plot['Mood'],
                'Social Activity': df_plot['Social']
            }).set_index('Date')

            st.line_chart(chart_data)

            # Add text explanation for change point
            if is_declining:
                st.markdown("<p style='color: #888;'>**Note:** Subtle decline simulated in this pattern.</p>", unsafe_allow_html=True)

        elif selected_metric == "Sleep Hours":
            st.line_chart(df.set_index('Date')['Sleep'])
            st.markdown("<p style='color: #888;'>**
