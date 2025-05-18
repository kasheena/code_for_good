import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import altair as alt

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
    st.set_page_config(page_title="Mental Health Early Warning System", layout="wide")
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

    st.title("MindBloom: Mental Wellness Tracker")  # More engaging title

    # Introduction with a Gen-Z tone
    st.markdown(
        "Hey there! 👋 Welcome to MindBloom, your personal space to cultivate a healthy mind. "
        "Think of this as your digital self-care garden. Let's grow together! 🌱"
    )

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Reflect & Grow", "Bloom Tracker", "About MindBloom"]) #Tab names

    # Tab 1: Text Analysis
    with tab1:
        st.header("Reflect on Your Day")  # More engaging header

        # User can select a sample or enter their own text
        use_sample = st.checkbox("Need a writing prompt?", value=True) # Changed Checkbox text

        if use_sample:
            sample_idx = st.selectbox(
                "Select a prompt:", # Changed dropdown text
                range(len(sample_entries)),
                format_func=lambda i: f"Prompt {i+1}: {sample_entries[i]['risk_level']} focus" # changed format
            )
            text_input = sample_entries[sample_idx]["text"]
            text_input = st.text_area("Your thoughts:", value=text_input, height=150) # Changed text area label
        else:
            text_input = st.text_area(
                "Write your journal entry, rant, or share anything on your mind:", # Changed placeholder
                value="",
                height=150,
                placeholder="Type here..."
            )

        if st.button("Analyze My Reflection"): # changed button text
            with st.spinner("Analyzing your reflection..."): # Changed spinner text
                # Analyze the text
                analysis_result = analyze_text(text_input)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Your Vibe Check") # Changed subheader
                    sentiment = analysis_result['sentiment']

                    st.markdown("**Overall Mood Score:**") # Changed text
                    sentiment_value = sentiment['compound']
                    st.progress((sentiment_value + 1) / 2)
                    st.markdown(f"Mood: **{sentiment_value:.2f}** ({'Positive' if sentiment_value > 0 else 'Negative'})") # Changed text

                    st.markdown("Deeper Dive:") # Changed text
                    st.markdown(f"- Positive Vibes: **{sentiment['pos']:.2f}**") # Changed text
                    st.markdown(f"- Neutral Vibes: **{sentiment['neu']:.2f}**") # Changed text
                    st.markdown(f"- Negative Vibes: **{sentiment['neg']:.2f}**") # Changed text

                with col2:
                    st.subheader("Wellness Focus") # Changed subheader to wellness

                    # Show category scores
                    category_scores = analysis_result['category_scores']
                    if category_scores:
                        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
                        for category, score in sorted_categories:
                            st.markdown(f"**{category.title()}**")
                            st.progress(score)
                            st.markdown(f"Score: **{score:.2f}**")

                # Risk assessment
                st.subheader("Growth Potential") # Changed to growth potential
                risk_score = analysis_result['risk_score']

                # Display risk level - changed labels to be more positive and aligned with growth
                if risk_score < 30:
                    risk_label = "Blooming 🌱"
                    risk_color = "green"
                elif risk_score < 50:
                    risk_label = "Sprouting 🌿"
                    risk_color = "yellow"
                elif risk_score < 70:
                    risk_label = "Growing 🪴"
                    risk_color = "orange"
                else:
                    risk_label = "Needs some TLC 😢" #TLC = Tender loving care
                    risk_color = "red"

                col_risk1, col_risk2 = st.columns([1, 2])

                with col_risk1:
                    st.markdown(f"<h3 style='color: {risk_color};'>{risk_label}</h3>", unsafe_allow_html=True)
                    st.markdown(f"Growth Score: **{risk_score:.1f}/100**") # Changed text

                with col_risk2:
                    st.progress(risk_score / 100)
                    cols_labels = st.columns(4)
                    cols_labels[0].markdown("<p style='text-align: left;'>Blooming</p>", unsafe_allow_html=True)
                    cols_labels[1].markdown("<p style='text-align: center;'>Sprouting</p>", unsafe_allow_html=True)
                    cols_labels[2].markdown("<p style='text-align: center;'>Growing</p>", unsafe_allow_html=True)
                    cols_labels[3].markdown("<p style='text-align: right;'>Needs TLC</p>", unsafe_allow_html=True) # Changed

                # Recommendations based on risk level - changed to be more encouraging and less clinical
                st.subheader("Your Growth Plan")
                if risk_score < 30:
                    recommendations = [
                        "Keep nurturing your mind! You're doing great! 🌱",
                        "Continue your self-care routine.  Small steps, big impact.",
                        "Use the Bloom Tracker to monitor your progress and stay motivated."
                    ]
                elif risk_score < 50:
                    recommendations = [
                        "Let's add a little more sunshine! Try some mindfulness today. ☀️",
                        "Connect with your support system.  A little social boost can do wonders. 🫂",
                        "Ensure you're getting enough rest.  Sleep is food for the brain. 😴"
                    ]
                elif risk_score < 70:
                    recommendations = [
                        "You've got this!  Consider talking to a friend or mentor. 🗣️",
                        "Establish a consistent sleep schedule.  Your mind will thank you. ⏰",
                        "Get moving!  Even a short walk can clear your head. 🚶‍♀️",
                        "Practice gratitude.  Focus on the good stuff.  🙏"
                    ]
                else:
                    recommendations = [
                        "It's okay to ask for help.  Reach out to a professional.  We're here for you. 💖",
                        "Talk to someone you trust.  Sharing is caring. 🫂",
                        "Prioritize sleep, nutrition, and gentle exercise.  Be kind to your body. 💖",
                        "Use grounding techniques to stay present.  You are strong. 💪",
                        "Remember, you're not alone.  Support is available. 💖"
                    ]

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
        st.header("Your Bloom Tracker") # Changed header

        # Select profile type
        profile_type = st.radio(
            "Track your growth:", # Changed radio button text
            ["Show Stable Growth", "Show Growth with Challenges"] # Changed radio button options
        )

        # Generate appropriate data
        is_declining = profile_type == "Show Growth with Challenges" # changed variable name
        df = generate_time_series(days=30, declining=is_declining)

        # Display time series data
        st.subheader("30-Day Growth Journey") # Changed subheader

        # Select metric to display
        selected_metric = st.selectbox(
            "View your progress in:", # Changed selectbox text
            ["All Metrics", "Sleep Quality", "Mood Stability", "Social Connection"] # Changed selectbox options
        )

        if selected_metric == "All Metrics":
            # Normalize data for comparison
            df_plot = df.copy()
            df_plot['Sleep_norm'] = df_plot['Sleep'] / 8 * 100  # Normalize to 0-100 scale

            # Use Altair for more customization
            chart_data_melted = pd.melt(df_plot, id_vars=['Date'], value_vars=['Sleep_norm', 'Mood', 'Social'],
                                       var_name='Metric', value_name='Value')

            chart = alt.Chart(chart_data_melted).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Value:Q', title=None),
                color=alt.Color('Metric:N', title="Metric"),
                tooltip=['Date:T', 'Metric:N', 'Value:Q']
            ).properties(
                height=300
            )
            st.altair_chart(chart, use_container_width=True)

            # Add text explanation for change point
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Detected some challenges in your growth journey.</p>", unsafe_allow_html=True) # Changed text

        elif selected_metric == "Sleep Quality":
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Sleep:Q', title="Hours"),
                tooltip=['Date:T', 'Sleep:Q']
            ).properties(
                height=300
            )
            # Add a horizontal line for the concern threshold (6 hours)
            threshold = alt.Chart(pd.DataFrame({'y': [6]})).mark_rule(color='orange').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)

            st.markdown("<p style='color: orange;'>&#11044; Recommended sleep threshold (6 hours)</p>", unsafe_allow_html=True) #changed
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Potential dip in sleep quality detected.</p>", unsafe_allow_html=True) # Changed

        elif selected_metric == "Mood Stability":
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Mood:Q', title="Mood Score"),
                tooltip=['Date:T', 'Mood:Q']
            ).properties(
                height=300
            )
            # Add a horizontal line for the concern threshold (60/100)
            threshold = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule(color='orange').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown("<p style='color: orange;'>&#11044; Recommended mood stability threshold (60/100)</p>", unsafe_allow_html=True) #changed
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Potential dip in mood stability detected.</p>", unsafe_allow_html=True) # Changed

        else:  # Social Activity
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Social:Q', title="Social Connection"), # Changed
                tooltip=['Date:T', 'Social:Q']
            ).properties(
                height=300
            )
            # Add a horizontal line for the concern threshold (50/100)
            threshold = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(color='orange').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown("<p style='color: orange;'>&#11044; Recommended social connection threshold (50/100)</p>", unsafe_allow_html=True) #changed
            if is_declining:
                st.markdown("<p style='color: red;'>&#128680; Potential dip in social connection detected.</p>", unsafe_allow_html=True) # Changed

        # Add insight section
        st.subheader("Your Growth Insights") # Changed

        if is_declining:
            st.markdown("""
            **System Analysis:**

            - &#128680; **Detected some challenges** in your growth journey
            - &#128315; **Sleep pattern disruption** beginning approximately 2 weeks ago
            - &#128315; **Mood deterioration** showing significant correlation with sleep changes
            - &#128315; **Social engagement reduction** following initial wellbeing decline

            **Here's how we can nurture your growth:**

            - Schedule sleep assessment
            - Implement structured social reconnection plan
            - Consider mood support interventions
            """)
        else:
            st.markdown("""
            **System Analysis:**

            - &#128994; **Stable growth patterns** across all wellbeing indicators
            - &#128994; **Normal fluctuations** within expected ranges
            - &#128994; **No concerning trends** detected in the 30-day window

            **Keep thriving! Here are some tips:**

            - Continue your current wellness practices
            - Maintain regular check-ins with your Bloom Tracker
            - Consider adding new wellness activities to your routine
            """)

    # Tab 3: About the System
    with tab3:
        st.header("About MindBloom") # Changed

        st.markdown("""
        ### How It Works

        MindBloom uses a blend of technology and positive psychology to help you cultivate a healthy mind:

        1. **Reflect & Grow Component**
            - Sentiment analysis to understand your emotional state
            - Keyword identification to pinpoint specific areas for growth
            - Linguistic pattern recognition for deeper insights

        2. **Bloom Tracker**
            - Sleep quality tracking to monitor rest patterns
            - Mood fluctuation analysis for emotional stability
            - Social engagement monitoring for connection levels
            - Activity level assessment for overall wellbeing

        3. **Personalized Growth Algorithm**
            - Integration of multiple data points for a holistic view
            - Pattern detection over time to identify trends
            - Personalized baseline comparisons for tailored insights
            - Tailored recommendation generation for actionable steps
        """)

        st.subheader("Privacy & Ethics: Your Trust is Our Priority") # Changed

        st.markdown("""
        MindBloom is designed with your privacy and trust in mind:

        - **Privacy-preserving design**:  Your data stays with you.
        - **User control**:  You choose what to share and when.
        - **No data storage**:  Analysis happens in real-time.
        - **Transparent algorithms**:  We're open about how things work.
        - **Supportive, not diagnostic**:  We offer resources, not diagnoses.
        """)

        st.subheader("Potential Impact: Let's Grow Together") # Changed

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Individual Benefits:**
            - Earlier self-awareness
            - Access to tailored resources
            - Reduced stigma through technology
            - A positive and empowering approach to mental wellness
            """)

        with col2:
            st.markdown("""
            **Community Benefits:**
            - Promotes a culture of proactive mental wellness
            - Provides data-informed insights for community support
            - Encourages preventative care and reduces crisis situations
            - Fosters a more supportive and understanding community
            """)

if __name__ == "__main__":
    main()
