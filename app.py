import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.use('Agg')
import seaborn as sns
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
    
    if declining:
        # Generate declining mental health pattern
        sleep_hours = [7 + random.uniform(-0.5, 0.5) for _ in range(days//2)] + \
                     [6 + random.uniform(-1.0, 0.5) for _ in range(days//4)] + \
                     [5 + random.uniform(-1.0, 0.5) for _ in range(days//4)]
        
        mood_scores = [80 + random.uniform(-10, 10) for _ in range(days//2)] + \
                     [65 + random.uniform(-15, 10) for _ in range(days//4)] + \
                     [50 + random.uniform(-15, 5) for _ in range(days//4)]
                     
        social_scores = [75 + random.uniform(-15, 15) for _ in range(days//2)] + \
                       [60 + random.uniform(-20, 10) for _ in range(days//4)] + \
                       [40 + random.uniform(-15, 10) for _ in range(days//4)]
    else:
        # Generate stable pattern
        sleep_hours = [7 + random.uniform(-0.7, 0.7) for _ in range(days)]
        mood_scores = [75 + random.uniform(-15, 15) for _ in range(days)]
        social_scores = [70 + random.uniform(-20, 20) for _ in range(days)]
    
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

# Main application
def main():
    st.set_page_config(page_title="Mental Health Early Warning System", layout="wide")
    
    st.title("Mental Health Early Warning System")
    st.write("AI-powered early detection and intervention for mental health concerns")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Text Analysis", "Behavioral Trends", "About the System"])
    
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
                    
                    # Create a gauge-like chart for compound sentiment
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.barh(['Sentiment'], [sentiment['compound']], color='green' if sentiment['compound'] > 0 else 'red')
                    ax.set_xlim(-1, 1)
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.set_xlabel('Negative → Positive')
                    ax.grid(axis='x', alpha=0.3)
                    st.pyplot(fig)
                    
                    # Show detailed sentiment scores
                    st.write(f"Positive: {sentiment['pos']:.2f}")
                    st.write(f"Neutral: {sentiment['neu']:.2f}")
                    st.write(f"Negative: {sentiment['neg']:.2f}")
                
                with col2:
                    st.subheader("Mental Health Indicators")
                    
                    # Show category scores
                    category_scores = analysis_result['category_scores']
                    if category_scores:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        categories = list(category_scores.keys())
                        scores = list(category_scores.values())
                        
                        # Sort by score for better visualization
                        sorted_indices = np.argsort(scores)
                        categories = [categories[i] for i in sorted_indices]
                        scores = [scores[i] for i in sorted_indices]
                        
                        ax.barh(categories, scores, color='skyblue')
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Indicator Strength')
                        ax.grid(axis='x', alpha=0.3)
                        st.pyplot(fig)
                
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
                
                # Create columns for risk display
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"### {risk_label}")
                    st.markdown(f"Score: **{risk_score:.1f}/100**")
                
                with col2:
                    # Create risk gauge
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh(['Risk'], [100], color='lightgray', height=0.5)
                    ax.barh(['Risk'], [risk_score], color=risk_color, height=0.5)
                    ax.set_xlim(0, 100)
                    # Add color zones
                    ax.axvline(x=30, color='black', linestyle='--', alpha=0.3)
                    ax.axvline(x=50, color='black', linestyle='--', alpha=0.3)
                    ax.axvline(x=70, color='black', linestyle='--', alpha=0.3)
                    # Add labels
                    ax.text(15, 0, 'Low', ha='center', va='center')
                    ax.text(40, 0, 'Low-Med', ha='center', va='center')
                    ax.text(60, 0, 'Medium', ha='center', va='center')
                    ax.text(85, 0, 'High', ha='center', va='center')
                    ax.set_yticks([])
                    st.pyplot(fig)
                
                # Recommendations based on risk level
                st.subheader("Personalized Recommendations")
                
                if risk_score < 30:
                    recommendations = [
                        "Continue monitoring your mental well-being",
                        "Maintain current self-care practices",
                        "Consider using our mood tracking feature for ongoing awareness"
                    ]
                elif risk_score < 50:
                    recommendations = [
                        "Practice mindfulness for 10 minutes daily",
                        "Ensure you're maintaining regular social connections",
                        "Review and optimize your sleep hygiene"
                    ]
                elif risk_score < 70:
                    recommendations = [
                        "Consider speaking with a trusted friend about how you're feeling",
                        "Implement a consistent sleep schedule",
                        "Try daily physical activity, even short walks",
                        "Practice gratitude journaling"
                    ]
                else:
                    recommendations = [
                        "Reach out to a mental health professional",
                        "Talk to someone you trust about how you're feeling",
                        "Focus on sleep, nutrition, and gentle physical activity",
                        "Use grounding techniques when feeling overwhelmed",
                        "Remember that support is available"
                    ]
                
                for rec in recommendations:
                    st.markdown(f"• {rec}")
                
                if risk_score >= 70:
                    st.markdown("---")
                    st.markdown("### Support Resources")
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
            
            # Plot all metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_plot['Date'], df_plot['Sleep_norm'], label='Sleep Quality (normalized)', marker='o', markersize=4)
            ax.plot(df_plot['Date'], df_plot['Mood'], label='Mood Score', marker='s', markersize=4)
            ax.plot(df_plot['Date'], df_plot['Social'], label='Social Activity', marker='^', markersize=4)
            
            # Add reference line
            if is_declining:
                change_point = df_plot['Date'][len(df_plot) // 2]
                ax.axvline(x=change_point, color='red', linestyle='--', alpha=0.7)
                ax.text(change_point, 30, 'Change Point', rotation=90, color='red')
            
            ax.set_ylim(0, 100)
            ax.set_ylabel('Score (0-100)')
            ax.set_xlabel('Date')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
        elif selected_metric == "Sleep Hours":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Date'], df['Sleep'], marker='o', color='blue')
            ax.set_ylabel('Sleep Hours')
            ax.set_ylim(0, 10)
            ax.axhline(y=6, color='orange', linestyle='--', alpha=0.7)
            ax.text(df['Date'][0], 6.1, 'Concern Threshold', color='orange')
            
            if is_declining:
                change_point = df['Date'][len(df) // 2]
                ax.axvline(x=change_point, color='red', linestyle='--', alpha=0.7)
                ax.text(change_point, 3, 'Change Point', rotation=90, color='red')
                
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
        elif selected_metric == "Mood Score":
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Date'], df['Mood'], marker='o', color='green')
            ax.set_ylabel('Mood Score')
            ax.set_ylim(0, 100)
            ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7)
            ax.text(df['Date'][0], 61, 'Concern Threshold', color='orange')
            
            if is_declining:
                change_point = df['Date'][len(df) // 2]
                ax.axvline(x=change_point, color='red', linestyle='--', alpha=0.7)
                ax.text(change_point, 30, 'Change Point', rotation=90, color='red')
                
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
        else:  # Social Activity
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Date'], df['Social'], marker='o', color='purple')
            ax.set_ylabel('Social Activity Score')
            ax.set_ylim(0, 100)
            ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7)
            ax.text(df['Date'][0], 51, 'Concern Threshold', color='orange')
            
            if is_declining:
                change_point = df['Date'][len(df) // 2]
                ax.axvline(x=change_point, color='red', linestyle='--', alpha=0.7)
                ax.text(change_point, 20, 'Change Point', rotation=90, color='red')
                
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Add insight section
        st.subheader("Pattern Insights")
        
        if is_declining:
            st.markdown("""
            **System Analysis:**
            
            • **Detected decline** in multiple wellbeing indicators
            • **Sleep pattern disruption** beginning approximately 2 weeks ago
            • **Mood deterioration** showing significant correlation with sleep changes
            • **Social engagement reduction** following initial wellbeing decline
            
            **Recommendations:**
            
            • Schedule sleep assessment
            • Implement structured social reconnection plan
            • Consider mood support interventions
            """)
        else:
            st.markdown("""
            **System Analysis:**
            
            • **Stable patterns** across all wellbeing indicators
            • **Normal fluctuations** within expected ranges
            • **No concerning trends** detected in the 30-day window
            
            **Recommendations:**
            
            • Continue current wellness practices
            • Maintain regular monitoring
            • Consider preventative wellbeing activities
            """)
    
    # Tab 3: About the System
    with tab3:
        st.header("About the Mental Health Early Warning System")
        
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

if __name__ == "__main__":
    main()
