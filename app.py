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

# --- Functions ---
def highlight_bias(text):
    for category, words in BIAS_RULES.items():
        for word in words:
            pattern = re.compile(rf'\b{re.escape(word)}\b', re.IGNORECASE)
            color = "red" if category == "male_coded" else "orange" if category == "exclusionary" else "blue"
            text = pattern.sub(
                lambda m: f"<span style='color:{color}; font-weight:bold'>{m.group(0)}</span>",
                text
            )
    return text

def calculate_bias_score(text):
    score = 0
    text_lower = text.lower()
    for category, words in BIAS_RULES.items():
        for word in words:
            if word.lower() in text_lower:
                score += 1 if category != "female_coded" else 0.5
    return min(10, score)

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

def extract_text_from_url(url):
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
        st.error(f"Error loading content: {str(e)}")
        return ""

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Job Ad Bias Detector", page_icon="🔍")

    st.sidebar.title("About")
    st.sidebar.markdown("""
    This tool detects biased language in job postings that may discourage diverse applicants.

    - **Red**: Male-coded terms  
    - **Blue**: Female-coded terms  
    - **Orange**: Exclusionary terms
    """)

    st.title("🔍 Job Ad Bias Detector")
    st.markdown("Paste a job description, upload a PDF, or enter a URL to analyze for biased language.")

    input_method = st.radio("Select input method:", ("Text", "PDF Upload", "URL"))

    # Clear job_desc if input method changes
    if "last_input_method" not in st.session_state:
        st.session_state["last_input_method"] = input_method
    elif st.session_state["last_input_method"] != input_method:
        st.session_state["job_desc"] = ""
        st.session_state["last_input_method"] = input_method

    job_desc = ""

    if input_method == "Text":
        job_desc = st.text_area("Paste job description here:", st.session_state.get("job_desc", ""), height=300)
        st.session_state["job_desc"] = job_desc

    elif input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a job description PDF", type="pdf")
        if uploaded_file:
            job_desc = extract_text_from_pdf(uploaded_file)
            st.session_state["job_desc"] = job_desc
            st.success("PDF content loaded!")

    elif input_method == "URL":
        url = st.text_input("Enter URL (supports job pages or raw GitHub .txt):")
        if st.button("Load Content"):
            with st.spinner("Loading content..."):
                job_desc = extract_text_from_url(url)
                if job_desc:
                    st.session_state["job_desc"] = job_desc
                    st.success("Content loaded successfully!")
        job_desc = st.session_state.get("job_desc", "")

    if job_desc and st.button("Analyze"):
        with st.spinner("Detecting biases..."):
            highlighted = highlight_bias(job_desc)
            st.markdown("### Highlighted Job Description")
            st.markdown(highlighted, unsafe_allow_html=True)

            bias_score = calculate_bias_score(job_desc)
            st.progress(bias_score / 10, text=f"Bias Score: {bias_score}/10 (lower is better)")

            st.subheader("Suggestions")
            if bias_score >= 7:
                st.error("⚠️ Highly biased language detected. Consider rewriting this job ad.")
            elif bias_score >= 4:
                st.warning("⚠️ Moderate bias detected. Some terms may discourage applicants.")
            else:
                st.success("✅ Relatively neutral language detected.")

            st.markdown("**Common fixes:**")
            st.markdown("- 'Rockstar developer' → 'Skilled developer'")
            st.markdown("- 'Dominant personality' → 'Leadership skills'")
            st.markdown("- 'Young and energetic' → 'Enthusiastic'")

    st.markdown("---")
    st.markdown("Built with ♥ using Streamlit")

if __name__ == "__main__":
    main()
