import streamlit as st
from openai import OpenAI
import pandas as pd
import plotly.express as px

st.title("Cover Letter Generator✉️")
st.markdown("This app tailors a coverletter based on the provided job description and the candidate's resume.")

# OpenAI API Key input
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    type="password", 
    help="You can find your API key at https://platform.openai.com/account/api-keys"
)


def coverletter_generator_openai(job_description_text,cv_text, sample_coverletter):
    """
    Classify the sentiment of a customer review using OpenAI's GPT-4o model.
    Parameters:
        review_text (str): The customer review text to be classified.
    Returns:
        str: The sentiment classification of the review as a single word, "positive", "negative", or "neutral".
    """
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Act like an experienced hiring manager, based on the job description below, help me pick 20 key words and skills that a candidate should include in his resume to pass the ATS (ranking the most important key words/skills starting from top). Also generate 3 main problems the hiring manager is looking to solve. Job description:{job_description_text}.
        
        Create a cover letter for the candidate, based on the job description:{job_description_text} and the candidate's resume:{cv_text}, to address the hiring manager's problems. Please use the format of the sample cover letter:{sample_coverletter}. The cover letter should fit concisely into one A4 size page.

        Also, act like a professional career coach, advise how the CV could be improved. Generate an improved CV.

        Structure your response using these exact markers:
        ---ATS KEYWORDS---
        (keywords and hiring manager problems here)
        ---COVER LETTER---
        (cover letter here)
        ---REVISED CV---
        (revised CV here)
        '''
    # prompt = f'''
        
        # Classify the following customer review. 
        # State your answer
        # as a single word, "positive", 
        # "negative" or "neutral":

        # {review_text}
        # '''

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    ) 

    return completion.choices[0].message.content


# Example TXT format
with st.expander("📋 See example job description format"):
    st.markdown("Your job description should be a plain `.txt` file. Here's a simple example:")
    st.code("""Job Title: Data Analyst
Company: St. Luke's Hospital

About the Role:
We are looking for a detail-oriented Data Analyst to join our analytics team.
The candidate will support data-driven decision making across clinical operations.

Responsibilities:
- Analyse large datasets to identify trends and insights
- Build and maintain dashboards and reports
- Collaborate with stakeholders to understand data needs

Requirements:
- Proficiency in SQL, Python, or R
- Experience with data visualisation tools (e.g. Tableau, Power BI)
- Strong communication skills
- Healthcare experience is a plus
""", language="text")
    st.caption("Copy this format into a .txt file and upload it below.")

# PDF file uploader (job description)
uploaded_job_file = st.file_uploader(
    "Upload a txt file for the job", 
    type=["txt"])

# Example CV format
with st.expander("📋 See example CV format"):
    st.markdown("Your CV should be a plain `.txt` file. Here's a simple example:")
    st.code('TAN CHUN WEI\nPhone: +65 9123 4567 | Email: chunwei@email.com | LinkedIn: linkedin.com/in/chunwei\n\nPROFESSIONAL SUMMARY\nData Analyst with 3+ years of experience in healthcare analytics, dashboard development,\nand stakeholder reporting. Skilled in translating complex data into actionable insights.\n\nKEY SKILLS\n- Data Analysis: SQL, Python, Excel\n- Visualisation: Tableau, Power BI\n- Reporting & Dashboards: automated reporting pipelines\n- Stakeholder Engagement: cross-functional collaboration\n- Healthcare domain knowledge\n\nPROFESSIONAL EXPERIENCE\nABC Healthcare Group | Data Analyst | Jan 2022 - Present\n- Built dashboards tracking KPIs across 5 clinical departments\n- Automated monthly reporting, reducing manual effort by 60%\n- Collaborated with ops and finance teams to support data-driven decisions\n\nXYZ Analytics Pte Ltd | Junior Analyst | Jun 2020 - Dec 2021\n- Conducted data cleaning and analysis for retail and healthcare clients\n- Produced weekly performance reports for senior stakeholders\n\nEDUCATION\nBSc Business Analytics | National University of Singapore | 2020\n\nCERTIFICATIONS\n- Google Data Analytics Certificate\n- Tableau Desktop Specialist\n', language="text")
    st.caption("Copy this format into a .txt file and upload it below.")

# PDF file uploader
uploaded_cv_file = st.file_uploader(
    "Upload a txt file for your CV", 
    type=["txt"])

# CSV file uploader
# uploaded_file = st.file_uploader(
#     "Upload a CSV file with restaurant reviews", 
#     type=["csv"])


# Initialise session state
if "ats_section" not in st.session_state:
    st.session_state.ats_section = None
if "cover_letter_section" not in st.session_state:
    st.session_state.cover_letter_section = None
if "revised_cv_section" not in st.session_state:
    st.session_state.revised_cv_section = None

# Parse sections using markers
def extract_section(text, start_marker, end_marker=None):
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    if end_marker:
        end = text.find(end_marker, start)
        return text[start:end].strip() if end != -1 else text[start:].strip()
    return text[start:].strip()

# Generate and Reset buttons side by side
st.markdown("---")
col_gen, col_reset = st.columns([3, 1])
with col_gen:
    generate_clicked = st.button("🚀 Generate Cover Letter & Revised CV", use_container_width=True)
with col_reset:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    st.session_state.ats_section = None
    st.session_state.cover_letter_section = None
    st.session_state.revised_cv_section = None
    st.rerun()

if generate_clicked:
    # Validate all inputs and show errors
    errors = []
    if not openai_api_key:
        errors.append("🔑 OpenAI API key is missing. Please enter it in the sidebar.")
    if uploaded_job_file is None:
        errors.append("📄 Job description file is missing. Please upload a TXT file.")
    if uploaded_cv_file is None:
        errors.append("📋 CV file is missing. Please upload a TXT file.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        cv_text = uploaded_cv_file.read().decode("utf-8", errors="replace")
        job_text = uploaded_job_file.read().decode("utf-8", errors="replace")

        with open('sample_coverletter.txt') as f:
            sample_text = f.read()

        with st.spinner("Generating your cover letter and revised CV..."):
            full_response = coverletter_generator_openai(job_text, cv_text, sample_text)

        st.session_state.ats_section = extract_section(full_response, "---ATS KEYWORDS---", "---COVER LETTER---")
        st.session_state.cover_letter_section = extract_section(full_response, "---COVER LETTER---", "---REVISED CV---")
        st.session_state.revised_cv_section = extract_section(full_response, "---REVISED CV---")

# Display results if they exist in session state
if st.session_state.ats_section:
    st.subheader("🔑 ATS Keywords & Hiring Manager Problems")
    st.write(st.session_state.ats_section)

    st.subheader("✉️ Cover Letter")
    st.write(st.session_state.cover_letter_section)

    st.subheader("📄 Revised CV")
    st.write(st.session_state.revised_cv_section)

    # Download buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ Download Cover Letter",
            data=st.session_state.cover_letter_section.encode("utf-8"),
            file_name="cover_letter.txt",
            mime="text/plain"
        )
    with col2:
        st.download_button(
            label="⬇️ Download Revised CV",
            data=st.session_state.revised_cv_section.encode("utf-8"),
            file_name="revised_cv.txt",
            mime="text/plain"
        )

    # # Create 3 columns to display the 3 metrics
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     # Show the number of positive reviews and the percentage
    #     positive_count = sentiment_counts.get("Positive", 0)
    #     st.metric("Positive", 
    #               positive_count, 
    #               f"{positive_count / len(reviews_df) * 100:.2f}%")
    
    # with col2:
    #     # Show the number of neutral reviews and the percentage
    #     neutral_count = sentiment_counts.get("Neutral", 0)
    #     st.metric("Neutral", 
    #               neutral_count, 
    #               f"{neutral_count / len(reviews_df) * 100:.2f}%")
    
    # with col3:
    #     # Show the number of negative reviews and the percentage
    #     negative_count = sentiment_counts.get("Negative", 0)
    #     st.metric("Negative", 
    #               negative_count, 
    #               f"{negative_count / len(reviews_df) * 100:.2f}%")
        
    # # Display pie chart
    # fig = px.pie(
    #     values=sentiment_counts.values, 
    #     names=sentiment_counts.index, 
    #     title='Sentiment Distribution'
    # )
    # st.plotly_chart(fig)
