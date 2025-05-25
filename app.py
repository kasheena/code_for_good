import streamlit as st
import PyPDF2
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import re

# --- Constants ---
BIAS_RULES = {
    "male_coded": ["aggressive", "ambitious", "assertive", "challenge", "competitive", 
                  "confident", "courageous", "decide", "decision", "decisive", 
                  "dominant", "dominance", "driven", "independent", "individual", 
                  "lead", "objective", "outspoken", "persist", "principle", 
                  "superior", "tough"],
    "female_coded": ["agree", "affectionate", "child", "cheer", "collaborate", 
                    "commit", "communal", "compassion", "connect", "considerate", 
                    "cooperate", "depend", "emotion", "empath", "feel", "flatterable", 
                    "gentle", "honest", "interpersonal", "kind", "kinship", "loyal", 
                    "modesty", "nag", "nurtur", "pleasant", "polite", "quiet", "respon", 
                    "sensitiv", "submissive", "support", "sympath", "tender", "together", 
                    "trust", "understand", "warm", "whin", "yield"],
    "exclusionary": ["rockstar", "ninja", "guru", "wizard", "young", "fresh", 
                    "recent grad", "digital native", "fast-paced", "work hard, play hard"]
}

# Color mapping for bias categories
BIAS_COLORS = {
    "male_coded": "#FF4B4B",      # Red
    "female_coded": "#1E88E5",    # Blue  
    "exclusionary": "#FF8C00"     # Orange
}

# --- Functions ---
def highlight_bias(text):
    """Add HTML spans to highlight biased terms with improved word boundary detection"""
    highlighted_text = text
    found_terms = []
    
    for category, words in BIAS_RULES.items():
        color = BIAS_COLORS[category]
        for word in words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.finditer(pattern, highlighted_text, re.IGNORECASE)
            
            for match in matches:
                matched_word = match.group()
                if matched_word.lower() not in [term.lower() for term, _ in found_terms]:
                    found_terms.append((matched_word, category))
                    highlighted_text = highlighted_text.replace(
                        matched_word, 
                        f"<span style='background-color:{color}20;color:{color};font-weight:bold;padding:2px 4px;border-radius:3px'>{matched_word}</span>"
                    )
    
    return highlighted_text, found_terms

def calculate_bias_score(text):
    """Calculate bias score with improved weighting (0-10 scale)"""
    male_count = 0
    female_count = 0
    exclusionary_count = 0
    
    text_lower = text.lower()
    
    for word in BIAS_RULES["male_coded"]:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            male_count += 1
    
    for word in BIAS_RULES["female_coded"]:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            female_count += 1
            
    for word in BIAS_RULES["exclusionary"]:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            exclusionary_count += 1
    
    # Weighted scoring: exclusionary terms are most problematic
    score = (male_count * 1.0) + (female_count * 0.7) + (exclusionary_count * 1.5)
    
    return min(10, score), {
        "male_coded": male_count,
        "female_coded": female_count,
        "exclusionary": exclusionary_count
    }

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF with error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_url(url):
    """Improved web scraping for job ads with better error handling"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            element.decompose()
        
        # Try to find job-specific content first
        job_content = soup.find(['div', 'section'], class_=re.compile(r'job|description|content', re.I))
        if job_content:
            text = job_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
        
        return text[:5000]  # Limit to first 5000 characters to avoid overwhelming UI
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Error processing content: {str(e)}")
        return ""

def get_recommendations(bias_counts):
    """Generate specific recommendations based on detected biases"""
    recommendations = []
    
    if bias_counts["male_coded"] > 0:
        recommendations.append("🔴 **Male-coded language detected**: Consider using more inclusive alternatives that don't emphasize traditionally masculine traits.")
    
    if bias_counts["female_coded"] > 0:
        recommendations.append("🔵 **Female-coded language detected**: While not inherently negative, be aware these terms might reinforce stereotypes.")
    
    if bias_counts["exclusionary"] > 0:
        recommendations.append("🟠 **Exclusionary terms detected**: These terms may discourage diverse candidates from applying.")
    
    if not any(bias_counts.values()):
        recommendations.append("✅ **Good job!** Your job posting appears to use inclusive language.")
    
    return recommendations

# --- Streamlit App ---
st.set_page_config(
    page_title="Job Ad Bias Detector", 
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.bias-legend {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.color-box {
    width: 20px;
    height: 20px;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 Job Ad Bias Detector")
st.markdown("**Analyze job postings for biased language that may discourage diverse applicants**")

# Sidebar
with st.sidebar:
    st.title("ℹ️ About This Tool")
    st.markdown("""
    This tool identifies potentially biased language in job postings based on research showing certain words may discourage diverse candidates.
    
    **Legend:**
    """)
    
    # Color legend in sidebar
    st.markdown(f"""
    <div class="legend-item">
        <div class="color-box" style="background-color: {BIAS_COLORS['male_coded']};"></div>
        <span>Male-coded terms</span>
    </div>
    <div class="legend-item">
        <div class="color-box" style="background-color: {BIAS_COLORS['female_coded']};"></div>
        <span>Female-coded terms</span>
    </div>
    <div class="legend-item">
        <div class="color-box" style="background-color: {BIAS_COLORS['exclusionary']};"></div>
        <span>Exclusionary terms</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Quick Tips:**")
    st.markdown("• Focus on skills and qualifications")
    st.markdown("• Use neutral, inclusive language")
    st.markdown("• Avoid age-related terms")
    st.markdown("• Consider accessibility requirements")

# Demo data
DEMO_TEXTS = {
    "Select a demo...": "",
    "Tech (Highly Biased)": "We need a ROCKSTAR Python ninja who can dominate the codebase! Must be aggressive in code reviews, work in a fast-paced environment, and have that young, driven mentality. Only digital natives need apply!",
    "Marketing (Moderately Biased)": "Seeking a compassionate marketing specialist to nurture client relationships. Must be pleasant, supportive, and understand emotional connections. We value collaboration and interpersonal skills.",
    "Finance (Neutral)": "Financial Analyst position available. Requirements: Bachelor's degree in Finance, 3+ years experience, proficiency in Excel and SQL, strong analytical skills, attention to detail.",
    "Sales (Mixed Bias)": "Ambitious sales professional needed! Must be confident and decisive in closing deals, while maintaining honest and considerate client relationships. Leadership experience preferred."
}

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Input method selection
    input_method = st.radio("Choose input method:", ("Text Input", "PDF Upload", "URL Scraping"), horizontal=True)
    
    job_desc = ""
    
    if input_method == "Text Input":
        demo_selection = st.selectbox("Try a demo example:", list(DEMO_TEXTS.keys()), index=0)
        job_desc = st.text_area(
            "Paste your job description here:", 
            value=DEMO_TEXTS[demo_selection],
            height=250,
            placeholder="Enter the job posting text you want to analyze..."
        )
        
    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF job posting", type=["pdf"])
        if uploaded_file:
            with st.spinner("Extracting text from PDF..."):
                job_desc = extract_text_from_pdf(uploaded_file)
                if job_desc:
                    st.success(f"✅ Extracted {len(job_desc)} characters from PDF")
                    with st.expander("Preview extracted text"):
                        st.text(job_desc[:500] + "..." if len(job_desc) > 500 else job_desc)
                        
    elif input_method == "URL Scraping":
        url = st.text_input("Enter the job posting URL:", placeholder="https://example.com/job-posting")
        if st.button("🔗 Scrape Content", type="secondary"):
            if url and url.startswith(('http://', 'https://')):
                with st.spinner("Fetching content from URL..."):
                    job_desc = extract_text_from_url(url)
                    if job_desc:
                        st.success(f"✅ Scraped {len(job_desc)} characters from URL")
                        with st.expander("Preview scraped content"):
                            st.text(job_desc[:500] + "..." if len(job_desc) > 500 else job_desc)
            else:
                st.error("Please enter a valid URL starting with http:// or https://")

with col2:
    st.markdown("### 📊 Analysis Results")
    results_placeholder = st.empty()

# Analysis button and results
if st.button("🔍 Analyze for Bias", type="primary", use_container_width=True):
    if job_desc.strip():
        with st.spinner("Analyzing job posting for biased language..."):
            # Perform analysis
            highlighted_text, found_terms = highlight_bias(job_desc)
            bias_score, bias_counts = calculate_bias_score(job_desc)
            
            # Display results in right column
            with results_placeholder.container():
                # Score display
                score_color = "red" if bias_score >= 7 else "orange" if bias_score >= 4 else "green"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Bias Score</h4>
                    <h2 style="color: {score_color};">{bias_score:.1f}/10</h2>
                    <small>Lower scores indicate more inclusive language</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Breakdown
                st.markdown("**Detection Breakdown:**")
                st.write(f"🔴 Male-coded: {bias_counts['male_coded']}")
                st.write(f"🔵 Female-coded: {bias_counts['female_coded']}")
                st.write(f"🟠 Exclusionary: {bias_counts['exclusionary']}")
            
            # Main results area
            st.markdown("---")
            st.subheader("🎯 Highlighted Text")
            
            if found_terms:
                st.markdown("**Found biased terms:**")
                for term, category in found_terms:
                    color = BIAS_COLORS[category]
                    st.markdown(f"<span style='color: {color}; font-weight: bold;'>• {term}</span> ({category.replace('_', ' ')})", unsafe_allow_html=True)
                st.markdown("---")
            
            # Display highlighted text
            st.markdown(highlighted_text, unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            recommendations = get_recommendations(bias_counts)
            for rec in recommendations:
                st.markdown(rec)
            
            # Improvement suggestions
            if bias_score >= 4:
                st.subheader("🔧 Suggested Improvements")
                st.markdown("""
                **Common replacements:**
                - 'Rockstar/Ninja/Guru' → 'Skilled/Experienced/Expert'
                - 'Aggressive' → 'Proactive/Results-driven'
                - 'Dominant' → 'Leadership-oriented'
                - 'Young and energetic' → 'Enthusiastic'
                - 'Digital native' → 'Tech-savvy'
                - 'Work hard, play hard' → 'Results-focused with work-life balance'
                """)
    else:
        st.warning("⚠️ Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Built with ❤️ by Kasheena Mulla | 
    <a href="https://streamlit.io" target="_blank">Powered by Streamlit</a>
</div>
""", unsafe_allow_html=True)
