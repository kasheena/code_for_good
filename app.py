import streamlit as st
from transformers import pipeline
import PyPDF2
from io import BytesIO
import requests
from bs4 import BeautifulSoup

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

# --- Functions ---
def highlight_bias(text):
    """Add HTML spans to highlight biased terms"""
    for category, words in BIAS_RULES.items():
        for word in words:
            if word.lower() in text.lower():
                color = "red" if category == "male_coded" else "orange" if category == "exclusionary" else "blue"
                text = text.replace(word, f"<span style='color:{color};font-weight:bold'>{word}</span>")
                text = text.replace(word.lower(), f"<span style='color:{color};font-weight:bold'>{word.lower()}</span>")
    return text

def calculate_bias_score(text):
    """Calculate simple bias score (0-10)"""
    score = 0
    text_lower = text.lower()
    for category in BIAS_RULES:
        for word in BIAS_RULES[category]:
            if word.lower() in text_lower:
                score += 1 if category != "female_coded" else 0.5  # Weight male-coded/exclusionary higher
    return min(10, score)  # Cap at 10

def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    return " ".join([page.extract_text() for page in pdf_reader.pages])

def extract_text_from_url(url):
    """Basic web scraping for job ads"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join([p.get_text() for p in soup.find_all('p')])
    except:
        return ""

# --- Streamlit App ---
st.set_page_config(page_title="Job Ad Bias Detector", page_icon="🔍")

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("""
This tool detects biased language in job postings that may discourage diverse applicants.
- **Red**: Male-coded terms
- **Blue**: Female-coded terms
- **Orange**: Exclusionary terms
""")

# Main UI
st.title("🔍 Job Ad Bias Detector")
st.markdown("Paste a job description, upload a PDF, or enter a URL to analyze for biased language.")

input_method = st.radio("Input method:", ("Text", "PDF Upload", "URL"))

job_desc = ""
if input_method == "Text":
    job_desc = st.text_area("Paste job description here:", height=300)
elif input_method == "PDF Upload":
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        job_desc = extract_text_from_pdf(uploaded_file)
elif input_method == "URL":
    url = st.text_input("Enter job posting URL:")
    if url:
        job_desc = extract_text_from_url(url)

if st.button("Analyze") and job_desc:
    with st.spinner("Detecting biases..."):
        # Highlight and display text
        highlighted_text = highlight_bias(job_desc)
        st.markdown(highlighted_text, unsafe_allow_html=True)
        
        # Calculate score
        bias_score = calculate_bias_score(job_desc)
        st.progress(bias_score / 10, text=f"Bias Score: {bias_score}/10 (lower is better)")
        
        # Show suggestions
        st.subheader("Suggestions")
        if bias_score >= 7:
            st.error("⚠️ Highly biased language detected. Consider rewriting this job ad.")
        elif bias_score >= 4:
            st.warning("⚠️ Moderate bias detected. Some terms may discourage applicants.")
        else:
            st.success("✅ Relatively neutral language detected.")
        
        # Example replacements
        st.markdown("**Common fixes:**")
        st.markdown("- 'Rockstar developer' → 'Skilled developer'")
        st.markdown("- 'Dominant personality' → 'Leadership skills'")
        st.markdown("- 'Young and energetic' → 'Enthusiastic'")

# Footer
st.markdown("---")
st.markdown("Built with ♥ using [Streamlit](https://streamlit.io) | [Hugging Face](https://huggingface.co)")
