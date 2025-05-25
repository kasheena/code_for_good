import streamlit as st
import PyPDF2
from io import BytesIO
import requests
from bs4 import BeautifulSoup
import re
import asyncio

# --- Fix event loop ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Bias Rules ---
BIAS_RULES = {
    "male_coded": [
        "ambitious", "competitive", "confident", "independent", "analytical",
        "technical", "objective", "strategic", "leader", "challenge",
        "decisive", "bold", "entrepreneurial", "assertive", "drive",
        "dominant", "risk-taking", "competitive", "results-oriented",
        "self-reliant", "forceful", "logical", "tough-minded", "outspoken",
        "problem solver", "go-getter", "directive", "dominance", "command",
        "focused", "dominant", "vigorous", "persistent", "self-assured",
        "visionary", "trailblazer", "hands-on", "independent thinker",
        "take charge", "self-confident", "high-achiever", "decisiveness",
        "boldness", "fearless", "resilient", "champion", "competitive spirit"
    ],
    "female_coded": [
        "agree", "affectionate", "childlike", "cheer", "collaborate",
        "commit", "communal", "compassion", "connect", "considerate",
        "cooperate", "depend", "emotion", "empath", "feel", "flatterable",
        "gentle", "honest", "interpersonal", "kind", "kinship", "loyal",
        "modesty", "nag", "nurtur", "pleasant", "polite", "quiet", "respon",
        "sensitiv", "submissive", "support", "sympath", "tender", "together",
        "trust", "understand", "warm", "whin", "yield"
    ],
    "exclusionary": [
        "young and energetic", "digital native", "recent graduate", "native English speaker",
        "able-bodied", "must lift 50 pounds", "fast learner", "must own a car",
        "cultural fit", "like a family", "workaholic"
    ]
}

# --- Helper functions ---

def highlight_bias(text):
    """Highlight biased words with proper HTML spans and colors"""
    def replacer(match):
        word = match.group(0)
        # Find which category the word belongs to
        for category, words in BIAS_RULES.items():
            if word.lower() in words:
                color = "red" if category == "male_coded" else "blue" if category == "female_coded" else "orange"
                return f"<span style='color:{color}; font-weight:bold'>{word}</span>"
        return word

    # Build a regex pattern that matches any bias word (case insensitive)
    all_words = [re.escape(word) for words in BIAS_RULES.values() for word in words]
    pattern = re.compile(r'\b(' + '|'.join(all_words) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(replacer, text)


def calculate_bias_score(text):
    """Calculate bias score (0-10)"""
    score = 0
    text_lower = text.lower()
    for category, words in BIAS_RULES.items():
        weight = 1 if category != "female_coded" else 0.5  # male/exclusionary higher weight
        for word in words:
            if word.lower() in text_lower:
                score += weight
    return min(10, score)


def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    return " ".join([page.extract_text() or "" for page in pdf_reader.pages])


def extract_text_from_url(url):
    """Extract text content from URL"""
    try:
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        if url.endswith('.txt'):
            return response.text
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(["script", "style", "nav", "footer"]):
            element.decompose()
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        st.error(f"Error loading content: {e}")
        return ""

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="Job Ad Bias Detector", page_icon="🔍")

    st.sidebar.title("About")
    st.sidebar.markdown("""
    Detect biased language in job postings to promote inclusive hiring.
    - **Red**: Male-coded terms
    - **Blue**: Female-coded terms
    - **Orange**: Exclusionary terms
    """)

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
            st.success("PDF loaded successfully.")
    elif input_method == "URL":
        url = st.text_input("Enter URL (supports job postings or raw text files):")
        if st.button("Load Content") and url.strip():
            with st.spinner("Loading content..."):
                job_desc = extract_text_from_url(url)
                if job_desc:
                    st.success("Content loaded successfully.")
                else:
                    st.warning("Failed to load content.")

    if job_desc and st.button("Analyze"):
        with st.spinner("Analyzing for bias..."):
            highlighted = highlight_bias(job_desc)
            st.markdown(highlighted, unsafe_allow_html=True)

            bias_score = calculate_bias_score(job_desc)
            st.progress(bias_score / 10, text=f"Bias Score: {bias_score:.1f} / 10 (lower is better)")

            st.subheader("Suggestions")
            if bias_score >= 7:
                st.error("⚠️ Highly biased language detected. Consider revising this job ad.")
            elif bias_score >= 4:
                st.warning("⚠️ Moderate bias detected. Some terms may discourage diverse applicants.")
            else:
                st.success("✅ Language is relatively neutral and inclusive.")

            st.markdown("**Example fixes:**")
            st.markdown("- Replace 'rockstar' with 'skilled professional'")
            st.markdown("- Replace 'dominant' with 'strong leadership'")
            st.markdown("- Replace 'young and energetic' with 'enthusiastic and motivated'")

    st.markdown("---")
    st.markdown("Built with ♥ using Streamlit")

if __name__ == "__main__":
    main()
