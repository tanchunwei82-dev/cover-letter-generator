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

def classify_sentiment_openai(job_description_text,cv_text, sample_coverletter):
    """
    Classify the sentiment of a customer review using OpenAI's GPT-4o model.
    Parameters:
        review_text (str): The customer review text to be classified.
    Returns:
        str: The sentiment classification of the review as a single word, "positive", "negative", or "neutral".
    """
    client = OpenAI(api_key=openai_api_key)
    prompt = f'''
        Create a coverletter for the candidate, based on the job description:{job_description_text} and the candidate's resume:{cv_text}.

        Please use the format of the sample coverletter:{sample_coverletter}.
        The coverletter should fit concisely into one A4 size page.
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


# PDF file uploader (job description)
uploaded_job_file = st.file_uploader(
    "Upload a txt file for the job", 
    type=["txt"])

# PDF file uploader
uploaded_cv_file = st.file_uploader(
    "Upload a txt file for your CV", 
    type=["txt"])

# CSV file uploader
# uploaded_file = st.file_uploader(
#     "Upload a CSV file with restaurant reviews", 
#     type=["csv"])


# Once the user uploads a csv file:
if uploaded_job_file is not None  and uploaded_cv_file is not None: 
    # Read the file
    # reviews_df = pd.read_csv(uploaded_file)

    cv_text = uploaded_cv_file.read().decode("utf-8", errors="replace")
    job_text = uploaded_job_file.read().decode("utf-8", errors="replace")


    with open('sample_coverletter.txt') as f:
        sample_text = f.read()

    # # Check if the data has a text column
    # text_columns = reviews_df.select_dtypes(include="object").columns

    # if len(text_columns) == 0:
    #     st.error("No text columns found in the uploaded file.")

    # Show a dropdown menu to select the review column
    # review_column = st.selectbox(
    #     "Select the column with the customer reviews",
    #     text_columns
    # )

    # # Analyze the sentiment of the selected column
    # reviews_df["sentiment"] = reviews_df[review_column].apply(classify_sentiment_openai)
    
    # Display the sentiment distribution in metrics in 3 columns: Positive, Negative, Neutral
    # Make the strings in the sentiment column titled
    # reviews_df["sentiment"] = reviews_df["sentiment"].str.title()
    # sentiment_counts = reviews_df["sentiment"].value_counts()
    cover_letter = classify_sentiment_openai(job_text,cv_text,sample_text)
    st.write(cover_letter)
    # st.write(sentiment_counts)

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