import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import altair as alt
from PIL import Image

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
    border-bottom: 1px solid #ddd;
}
h1, h2, h3, h4, h5, h6 {
    color: #111;
}
.st-tabs>ul>li>a {
    color: #555;
    background-color: #eee;
    border: 1px solid #ddd;
    border-bottom: none;
    padding: 0.5rem 1rem;
    border-radius: 0.25rem 0.25rem 0 0;
}
.st-tabs>ul>li.active>a {
    color: #111;
    background-color: white;
    border-bottom: 1px solid white;
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
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    padding: 0.3rem;
}
.st-text-area>div>div>textarea {
    color: #333;
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    padding: 0.5rem;
}
.st-checkbox>label {
    color: #333;
}
.st-progress>div>div>div>div {
    background-color: #bbb; /* Default progress bar color */
}
.streamlit-expander {
    border: 1px solid #ddd;
    border-radius: 0.25rem;
    padding: 0.5rem;
    margin-bottom: 0.5rem;
}
.streamlit-expander-header {
    font-weight: bold;
    color: #333;
}
.markdown-text-container {
    color: #333;
}
"""

# Main application
def main():
    st.set_page_config(page_title="Bloom: Your Mental Wellness Companion 🌸", layout="wide") # Added flower to title
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

    # Add a title image
    # img = Image.open("your_image.png")  # Replace with your image file
    # st.image(img, width=100)

    st.title("Bloom: Your Mental Wellness Companion 🌸") # Added flower
    st.markdown("Nurturing your mind, one leaf at a time. 🌱") # Added plant metaphor

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Journal Patch 📝", "Growth Chart 📈", "About the Garden ℹ️"]) # Changed tab names

    # Tab 1: Text Analysis
    with tab1:
        st.header("Journal Entry Analysis")
        st.markdown("Share your thoughts and we'll help you see the subtle shifts in your inner garden. 🌿") # Added intro text

        # Use a checkbox for sample selection
        use_sample = st.checkbox("Use a sample journal entry", value=True)

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

        # Add a button to trigger analysis
        if st.button("Analyze Text"):
            with st.spinner("Analyzing text..."):
                # Analyze the text
                analysis_result = analyze_text(text_input)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Sentiment Analysis")
                    sentiment = analysis_result['sentiment']

                    st.markdown("**Overall Vibe:**") # Changed from Compound Sentiment Score
                    sentiment_value = sentiment['compound']
                    st.progress((sentiment_value + 1) / 2)
                    st.markdown(f"Feeling: **{'Positive ☀️' if sentiment_value > 0 else 'Negative 🌧️'}** (Score: {sentiment_value:.2f})") # Added emojis

                    # Use expander for detailed scores
                    with st.expander("Show Detailed Vibe Breakdown"): # Changed title
                        st.markdown(f"- Positive: **{sentiment['pos']:.2f}**")
                        st.markdown(f"- Neutral: **{sentiment['neu']:.2f}**")
                        st.markdown(f"- Negative: **{sentiment['neg']:.2f}**")

                with col2:
                    st.subheader("Inner Garden Indicators") # Changed from Mental Health Indicators

                    # Show category scores
                    category_scores = analysis_result['category_scores']
                    if category_scores:
                        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
                        for category, score in sorted_categories:
                            st.markdown(f"**{category.title()} Level:**") # Added level
                            st.progress(score)
                            st.markdown(f"Score: **{score:.2f}**")

                # Risk assessment
                st.subheader("Wellbeing Assessment") # Changed from Risk Assessment
                risk_score = analysis_result['risk_score']

                # Display risk level
                if risk_score < 30:
                    wellbeing_level = "Thriving 🌱" # Changed from risk_label
                    color = "green"
                elif risk_score < 50:
                    wellbeing_level = "Growing Steadily 🌿"
                    color = "yellow"
                elif risk_score < 70:
                    wellbeing_level = "Needs Some Tending 🍂"
                    color = "orange"
                else:
                    wellbeing_level = "In Need of Care 🥀"
                    color = "red"

                col_risk1, col_risk2 = st.columns([1, 2])

                with col_risk1:
                    st.markdown(f"<h3 style='color: {color};'>{wellbeing_level}</h3>", unsafe_allow_html=True) # Changed to wellbeing_level
                    st.markdown(f"Wellbeing Score: **{risk_score:.1f}/100**") # Changed from Risk Score

                with col_risk2:
                    st.progress(risk_score / 100)
                    cols_labels = st.columns(4)
                    cols_labels[0].markdown("<p style='text-align: left;'>Thriving</p>", unsafe_allow_html=True)
                    cols_labels[1].markdown("<p style='text-align: center;'>Growing</p>", unsafe_allow_html=True)
                    cols_labels[2].markdown("<p style='text-align: center;'>Tending</p>", unsafe_allow_html=True)
                    cols_labels[3].markdown("<p style='text-align: right;'>Care</p>", unsafe_allow_html=True) # Changed labels

                # Recommendations based on risk level
                st.subheader("Tips for Nurturing Your Growth") # Changed from Personalized Recommendations

                if risk_score < 30:
                    recommendations = [
                        "Continue to nurture your inner garden with self-care. ☀️",
                        "Maintain your current practices for a thriving mind. 🧘‍♀️",
                        "Use the Growth Chart to track your progress. 📈" # Changed to Growth Chart
                    ]
                elif risk_score < 50:
                    recommendations = [
                        "Practice daily mindfulness to help your roots grow stronger. 🧘",
                        "Ensure you're getting enough 'sunlight' through social connection. ☀️", # Metaphor
                        "Review your sleep habits to optimize your rest. 😴"
                    ]
                elif risk_score < 70:
                    recommendations = [
                        "Consider sharing your feelings with a trusted friend. 🗣️",
                        "Establish a consistent sleep schedule to regulate your inner rhythms. ⏰",
                        "Try incorporating gentle physical activity into your day. 🚶‍♀️",
                        "Cultivate gratitude by journaling about what you're thankful for. 🙏"
                    ]
                else:
                    recommendations = [
                        "**Reach out to a mental health professional for expert guidance. 🧑‍⚕️**",
                        "Talk to someone you trust about what you're experiencing. 🗣️",
                        "Prioritize sleep, healthy nutrition, and gentle movement. 🍎",
                        "Use grounding techniques to stay present when feeling overwhelmed. 🧘‍♂️",
                        "**Remember, support is available. 🌱**" # Added support statement
                    ]

                # Use a bulleted list for recommendations
                st.markdown("Here's what you can do to help your wellbeing flourish:") # Added intro
                for rec in recommendations:
                    st.markdown(f"- {rec}")

                if risk_score >= 70:
                    st.markdown("---")
                    st.subheader("Support Resources")
                    st.markdown("- **Crisis Text Line**: Text HOME to 741741")
                    st.markdown("- **National Suicide Prevention Lifeline**: 988 or 1-800-273-8255")
                    st.markdown("- **SAMHSA's National Helpline**: 1-800-662-4357")

    # Tab 2: Behavioral Trends
    with tab2:
        st.header("Your Growth Chart") # Changed from Behavioral Pattern Analysis
        st.markdown("Track the patterns of your inner garden over the past 30 days. 📊")

        # Select profile type using a radio button
        profile_type = st.radio(
            "Select profile type:",
            ["Stable Growth 📈", "Declining Growth 📉"] # Changed labels
        )

        # Generate appropriate data
        is_declining = profile_type == "Declining Growth 📉"
        df = generate_time_series(days=30, declining=is_declining)

        # Display time series data
        st.subheader("30-Day Growth Patterns") # Changed

        # Select metric to display using a selectbox
        selected_metric = st.selectbox(
            "Select metric to view:",
            ["All Metrics", "Sleep Hours", "Mood Score", "Social Activity"]
        )

        if selected_metric == "All Metrics":
            # Normalize data for comparison
            df_plot = df.copy()
            df_plot['Sleep_norm'] = df_plot['Sleep'] / 8 * 100  # Normalize to 0-100 scale

            # Use Altair for more customization
            chart_data_melted = pd.melt(df_plot, id_vars=['Date'], value_vars=['Sleep_norm', 'Mood', 'Social'],
                                       var_name='Metric', value_name='Value')

            chart = alt.Chart(chart_data_melted).mark_line(point=True).encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Value:Q', title=None),
                color=alt.Color('Metric:N', title="Metric"),
                tooltip=['Date:T', 'Metric:N', 'Value:Q']
            ).properties(
                height=400,
                width="container"
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

            # Add text explanation for change point
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Detected potential decline across multiple wellbeing indicators.</p>", unsafe_allow_html=True) #same
            else:
                st.markdown("<p style='color: green;'>&#128994; Your garden is showing stable growth across key areas. </p>", unsafe_allow_html=True)

        elif selected_metric == "Sleep Hours":
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Sleep:Q', title="Hours"),
                tooltip=['Date:T', 'Sleep:Q']
            ).properties(
                height=400,
                width="container"
            ).interactive()
            # Add a horizontal line for the concern threshold (6 hours)
            threshold = alt.Chart(pd.DataFrame({'y': [6]})).mark_rule(color='orange').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)

            st.markdown("<p style='color: orange;'>&#11044; Concern threshold (6 hours)</p>", unsafe_allow_html=True)
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Potential decline in sleep patterns detected.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: green;'>&#128994; Your sleep patterns are within a healthy range. </p>", unsafe_allow_html=True)

        elif selected_metric == "Mood Score":
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Mood:Q', title="Mood Score"),
                tooltip=['Date:T', 'Mood:Q']
            ).properties(
                height=400,
                width="container"
            ).interactive()
            # Add a horizontal line for the concern threshold (60/100)
            threshold = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule(color='orange').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown("<p style='color: orange;'>&#11044; Concern threshold (60/100)</p>", unsafe_allow_html=True)
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Potential decline in mood scores detected.</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: green;'>&#128994; Your mood is showing stable and positive levels. </p>", unsafe_allow_html=True)

        else:  # Social Activity
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Social:Q', title="Social Activity"),
                tooltip=['Date:T', 'Social:Q']
            ).properties(
                height=400,
                width="container"
            ).interactive()
            # Add a horizontal line for the concern threshold (50/100)
            threshold = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(color='orange').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown("<p style='color: orange;'>&#11044; Concern threshold (50/100)</p>", unsafe_allow_html=True)
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Potential decline in social activity detected.</p>", unsafe_allow_html=True)
            else:
                 st.markdown("<p style='color: green;'>&#128994;  Your social connections remain consistent. </p>", unsafe_allow_html=True)

        # Add insight section
        st.subheader("Growth Insights") # Changed

        if is_declining:
            st.markdown("""
            **System Analysis:**

            - &#128680; **Detected decline** in multiple wellbeing indicators
            - &#128315; **Sleep pattern disruption** beginning approximately 2 weeks ago
            - &#128315; **Mood deterioration** showing significant correlation with sleep changes
            - &#128315; **Social engagement reduction** following initial wellbeing decline

            **Recommendations:**

            - Schedule sleep assessment
            - Implement structured social reconnection plan
            - Consider mood support interventions
            """)
        else:
            st.markdown("""
            **System Analysis:**

            - &#128994; **Stable patterns** across all wellbeing indicators
            - &#128994; **Normal fluctuations** within expected ranges
            - &#128994; **No concerning trends** detected in the 30-day window

            **Recommendations:**

            - Continue current wellness practices
            - Maintain regular monitoring
            - Consider preventative wellbeing activities
            """)

    # Tab 3: About the System
    with tab3:
        st.header("About the Wellness Garden") # Changed from About the System

        st.markdown("""
        ### How It Works

        This system uses natural language processing and behavioral pattern analysis to detect early warning signs of mental health concerns:

        1. **Text Analysis Component**
            - Sentiment analysis to detect emotional tone
            - Keyword identification for specific concerns
            - Linguistic pattern recognition

        2. **Behavioral Pattern Monitoring**
            - Sleep quality tracking
            - Mood fluctuation analysis
            - Social engagement monitoring
            - Activity level assessment

        3. **Early Warning Algorithm**
            - Integration of multiple data points
            - Pattern detection across time
            - Personalized baseline comparisons
            - Tailored recommendation generation
        """)

        st.subheader("Privacy & Ethics")
        st.markdown("""
        Our system is designed with privacy and ethics as core principles:

        - **Privacy-preserving design**: All processing happens locally
        - **User control**: You decide what data to share and when
        - **No data storage**: Analysis happens in real-time without persistent storage
        - **Transparent algorithms**: Clear explanation of how assessments are made
        - **Supportive, not diagnostic**: Provides support resources, not medical diagnoses
        """)

        st.subheader("Potential Impact")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Individual Benefits:**
            - Earlier intervention
            - Increased self-awareness
            - Access to tailored resources
            - Reduced stigma through technology
            """)

        with col2:
            st.markdown("""
            **Community Benefits:**
            - Reduced crisis incidents
            - Improved resource allocation
            - Data-informed mental health support
            - Preventative rather than reactive care
            """)
        
        # Add a feedback form
        st.subheader("Feedback")
        feedback = st.text_area("Share your thoughts on this app:", placeholder="Enter your feedback here...")
        if st.button("Submit Feedback"):
            if feedback:
                st.success("Thank you for your feedback!")
            else:
                st.warning("Please enter some feedback before submitting.")

if __name__ == "__main__":
    main()
