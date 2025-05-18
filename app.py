import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import altair as alt

# Download necessary NLTK data (only needs to be done once)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Load a simple feel-good keyword dictionary
feel_good_keywords = {
    "sleep": ["sleep", "rest", "tired", "relaxed", "bed", "awake", "nap"],
    "happiness": ["happy", "joyful", "cheerful", "good", "smile", "laugh", "fun"],
    "calmness": ["calm", "peaceful", "relaxed", "serene", "quiet", "still", "tranquil"],
    "connection": ["friend", "family", "people", "together", "social", "talk", "connect"],
    "energy": ["energy", "active", "motivated", "lively", "enthusiastic", "passion", "enjoy"]
}

# Simple function to analyze text without external APIs
def analyze_text(text):
    text = text.lower()

    # Sentiment analysis
    sentiment = sia.polarity_scores(text)

    # Keyword matching
    category_scores = {}
    for category, keywords in feel_good_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        # Normalize score
        category_scores[category] = min(score / len(keywords) * 2, 1.0)

    # Calculate wellness score (simplified algorithm)
    wellness_score = (
        (sentiment['compound'] + 1) * 50 +  # Convert sentiment to 0-100 scale
        (category_scores.get('happiness', 0) * 15) +
        (category_scores.get('calmness', 0) * 12) +
        (category_scores.get('sleep', 0) * 10) +
        (category_scores.get('connection', 0) * 8) +
        (category_scores.get('energy', 0) * 5)
    )

    # Ensure wellness_score is in 0-100 range
    wellness_score = max(0, min(100, wellness_score))

    return {
        'sentiment': sentiment,
        'category_scores': category_scores,
        'wellness_score': wellness_score
    }

# Generate synthetic time series data
def generate_time_series(days=30, improving=True):
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()

    # Calculate lengths for each section to ensure equal array lengths
    section1 = days // 3
    section2 = days // 3
    section3 = days - section1 - section2  # Ensure all sections add up to exactly 'days'

    if improving:
        # Generate improving wellbeing pattern
        sleep_hours = [7 + random.uniform(-0.5, 0.5) for _ in range(section1)] + \
                      [7.5 + random.uniform(-0.5, 0.5) for _ in range(section2)] + \
                      [8 + random.uniform(-0.5, 0.5) for _ in range(section3)]
        mood_scores = [60 + random.uniform(-10, 10) for _ in range(section1)] + \
                      [70 + random.uniform(-10, 10) for _ in range(section2)] + \
                      [80 + random.uniform(-10, 10) for _ in range(section3)]
        social_scores = [50 + random.uniform(-10, 10) for _ in range(section1)] + \
                        [60 + random.uniform(-10, 10) for _ in range(section2)] + \
                        [70 + random.uniform(-10, 10) for _ in range(section3)]
    else:
        # Generate stable pattern
        sleep_hours = [7.5 + random.uniform(-0.7, 0.7) for _ in range(days)]
        mood_scores = [75 + random.uniform(-10, 10) for _ in range(days)]
        social_scores = [60 + random.uniform(-10, 10) for _ in range(days)]

    # Verify all lists have the same length
    assert len(dates) == len(sleep_hours) == len(mood_scores) == len(social_scores), "Array length mismatch"

    return pd.DataFrame({
        'Date': dates,
        'Sleep': sleep_hours,
        'Mood': mood_scores,
        'Social': social_scores
    })

# Sample text entries with pre-classified wellness levels
sample_entries = [
    {
        "text": "Feeling great today!  Had a good time with friends.",
        "wellness_level": "Great"
    },
    {
        "text": "A bit tired, but overall okay.  Relaxing at home.",
        "wellness_level": "Okay"
    },
    {
        "text": "Not feeling myself.  Haven't been sleeping well and feeling down.",
        "wellness_level": "Low"
    },
    {
        "text": "Things are really tough.  I feel very sad and alone.",
        "wellness_level": "Very Low"
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
    st.set_page_config(page_title="Feel Good App", layout="wide")
    st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

    st.title("Feel Good App")
    st.markdown("Your personal guide to feeling good")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Text Check-in", "Wellbeing Trends", "About"])

    # Tab 1: Text Analysis
    with tab1:
        st.header("Journal Check-in")

        # User can select a sample or enter their own text
        use_sample = st.checkbox("Use a sample entry", value=True)

        if use_sample:
            sample_idx = st.selectbox(
                "Select a sample journal entry:",
                range(len(sample_entries)),
                format_func=lambda i: f"Sample {i+1}: {sample_entries[i]['wellness_level']} "
            )
            text_input = sample_entries[sample_idx]["text"]
            text_input = st.text_area("How are you feeling today?", value=text_input, height=150)
        else:
            text_input = st.text_area(
                "Write a few lines about your day:",
                value="",
                height=150,
                placeholder="I'm feeling..."
            )

        if st.button("Check In") and text_input:
            with st.spinner("Analyzing your text..."):
                # Analyze the text
                analysis_result = analyze_text(text_input)

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Overall Mood")
                    sentiment = analysis_result['sentiment']

                    st.markdown("**Mood Score:**")
                    sentiment_value = sentiment['compound']
                    st.progress((sentiment_value + 1) / 2)  # Convert from -1,1 to 0,1 for progress bar
                    st.markdown(f"Score: **{sentiment_value:.2f}** ({'Good' if sentiment_value > 0 else 'Not so good'})")

                    st.markdown("Mood Details:")
                    st.markdown(f"- Positive: **{sentiment['pos']:.2f}**")
                    st.markdown(f"- Neutral: **{sentiment['neu']:.2f}**")
                    st.markdown(f"- Negative: **{sentiment['neg']:.2f}**")

                with col2:
                    st.subheader("Key Areas")

                    # Show category scores
                    category_scores = analysis_result['category_scores']
                    if category_scores:
                        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
                        for category, score in sorted_categories:
                            st.markdown(f"**{category.title()}**")
                            st.progress(score)
                            st.markdown(f"Score: **{score:.2f}**")

                # Wellness assessment
                st.subheader("Wellness Check")
                wellness_score = analysis_result['wellness_score']

                # Display wellness level
                if wellness_score < 30:
                    wellness_label = "Very Low"
                    wellness_color = "red"
                elif wellness_score < 50:
                    wellness_label = "Low"
                    wellness_color = "orange"
                elif wellness_score < 70:
                    wellness_label = "Okay"
                    wellness_color = "yellow"
                else:
                    wellness_label = "Great"
                    wellness_color = "green"

                col_wellness1, col_wellness2 = st.columns([1, 2])

                with col_wellness1:
                    st.markdown(f"<h3 style='color: {wellness_color};'>{wellness_label}</h3>", unsafe_allow_html=True)
                    st.markdown(f"Score: **{wellness_score:.1f}/100**")

                with col_wellness2:
                    st.progress(wellness_score / 100)
                    cols_labels = st.columns(4)
                    cols_labels[0].markdown("<p style='text-align: left;'>Very Low</p>", unsafe_allow_html=True)
                    cols_labels[1].markdown("<p style='text-align: center;'>Low</p>", unsafe_allow_html=True)
                    cols_labels[2].markdown("<p style='text-align: center;'>Okay</p>", unsafe_allow_html=True)
                    cols_labels[3].markdown("<p style='text-align: right;'>Great</p>", unsafe_allow_html=True)

                # Recommendations based on wellness level
                st.subheader("Suggestions for Today")

                if wellness_score < 30:
                    recommendations = [
                        "Reach out to a friend or family member for support.",
                        "Try some gentle stretching or deep breathing exercises.",
                        "Consider talking to a professional.",
                        "Remember, you are not alone."
                    ]
                elif wellness_score < 50:
                    recommendations = [
                        "Try a relaxing activity, like listening to music or taking a warm bath.",
                        "Make sure you're getting enough sleep.",
                        "Do something you enjoy, even for a short time.",
                        "Focus on small, achievable goals."
                    ]
                elif wellness_score < 70:
                    recommendations = [
                        "Engage in a hobby or creative activity.",
                        "Spend some time in nature.",
                        "Connect with loved ones.",
                        "Practice gratitude - write down a few things you appreciate."
                    ]
                else:
                    recommendations = [
                        "Keep up the good work!  Maintain your healthy habits.",
                        "Share your positive energy with others.",
                        "Try learning something new or taking on a fun challenge.",
                        "Reflect on your successes and celebrate them."
                    ]

                for rec in recommendations:
                    st.markdown(f"- {rec}")

                if wellness_score < 30:
                    st.markdown("---")
                    st.subheader("Where to Find Help")
                    st.markdown("- **Crisis Text Line**: Text HOME to 741741")
                    st.markdown("- **National Suicide Prevention Lifeline**: 988 or 1-800-273-8255")
                    st.markdown("- **SAMHSA's National Helpline**: 1-800-662-4357")

    # Tab 2: Behavioral Trends
    with tab2:
        st.header("Wellbeing Trends")

        # Select profile type
        profile_type = st.radio(
            "Show my trends as:",
            ["Stable", "Improving"]
        )

        # Generate appropriate data
        is_improving = profile_type == "Improving"
        df = generate_time_series(days=30, improving=is_improving)

        # Display time series data
        st.subheader("Your Last 30 Days")

        # Select metric to display
        selected_metric = st.selectbox(
            "Show me my:",
            ["All Metrics", "Sleep", "Mood", "Social Activity"]
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
            if is_improving:
                st.markdown("<p style='color: green;'>&#128300; Your wellbeing is improving!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: gray;'>&#128300; Your wellbeing is stable.</p>", unsafe_allow_html=True)

        elif selected_metric == "Sleep":
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

            st.markdown("<p style='color: orange;'>&#11044; Recommended (6 hours)</p>", unsafe_allow_html=True)
            if is_improving:
                st.markdown("<p style='color: green;'>&#128300; Your sleep is improving!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: gray;'>&#128300; Your sleep is stable.</p>", unsafe_allow_html=True)

        elif selected_metric == "Mood":
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Mood:Q', title="Mood Score"),
                tooltip=['Date:T', 'Mood:Q']
            ).properties(
                height=300
            )
            # Add a horizontal line for the concern threshold (60/100)
            threshold = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(color='green').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown("<p style='color: green;'>&#11044; Good Mood (70/100)</p>", unsafe_allow_html=True)
            if is_improving:
                st.markdown("<p style='color: green;'>&#128300; Your mood is improving!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: gray;'>&#128300; Your mood is stable.</p>", unsafe_allow_html=True)

        else:  # Social Activity
            chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Date:T', axis=alt.Axis(format="%Y-%m-%d")),
                y=alt.Y('Social:Q', title="Social Activity"),
                tooltip=['Date:T', 'Social:Q']
            ).properties(
                height=300
            )
            # Add a horizontal line for the concern threshold (50/100)
            threshold = alt.Chart(pd.DataFrame({'y': [60]})).mark_rule(color='green').encode(
                y=alt.Y('y'),
                tooltip=['y']
            )
            final_chart = chart + threshold
            st.altair_chart(final_chart, use_container_width=True)
            st.markdown("<p style='color: green;'>&#11044; Good Connection (60/100)</p>", unsafe_allow_html=True)
            if is_improving:
                st.markdown("<p style='color: green;'>&#128300; Your social connections are improving!</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: gray;'>&#128300; Your social connections are stable.</p>", unsafe_allow_html=True)

        # Add insight section
        st.subheader("What This Means For You")

        if is_improving:
            st.markdown("""
            **Overall:**
            - &#128300; We've spotted positive changes in your wellbeing over the past few weeks!

            **Details:**
            - &#128315; **Sleep:** You're getting more rest.
            - &#128315; **Mood:** You're feeling happier and more positive.
            - &#128315; **Social:** You're connecting more with others.

            **Keep it up!**
            - Continue practicing your self-care routine.
            - Stay connected with your support system.
            - Celebrate your progress!
            """)
        else:
            st.markdown("""
            **Overall:**
            - &#128994; Your wellbeing is stable, which is good!

            **Details:**
            - &#128994; **Sleep:** Your sleep patterns are consistent.
            - &#128994; **Mood:** You're feeling generally good.
            - &#128994; **Social:** You're maintaining your social connections.

            **Tips for staying balanced:**
            - Continue prioritizing sleep, healthy eating, and exercise.
            - Make time for activities you enjoy.
            - Stay connected with friends and family.
            """)

    # Tab 3: About the System
    with tab3:
        st.header("About Your Feel Good App")

        st.markdown("""
        ### How It Works

        This app helps you track and improve your overall wellbeing.  It looks at a few key things:

        1. **Text Check-in**
            -   We analyze your words to get a sense of your mood.
            -   We look for words related to happiness, calmness, and other positive feelings.

        2. **Wellbeing Trends**
            -   We track your sleep, mood, and social activity over time.
            -   This helps you see patterns and make positive changes.

        3.  **Personalized Suggestions**
            -   The app gives you ideas for things you can do to feel even better.
            -   Suggestions are tailored to your specific needs and progress.
        """)

        st.subheader("Your Privacy Matters")
        st.markdown("""
        We care about your privacy:

        -   **Your data stays with you:** Everything happens on your device.
        -   **You're in control:** You choose what you share.
        -   **We don't store your information:** Your check-ins are private.
        -   **We're here to support you:** This app is designed to help you feel your best.
        """)

        st.subheader("Why This App Can Help")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **For You:**
            -   Feel better, more often
            -   Learn more about yourself
            -   Get personalized support
            -   Reduce stress and improve mood
            """)

        with col2:
            st.markdown("""
            **For Our Community:**
            -   Promote positive mental wellbeing
            -   Encourage healthy habits
            -   Provide accessible support
            -   Help people thrive
            """)

if __name__ == "__main__":
    main()
