import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import nltk
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
# ---- Setup ----
nltk.download('stopwords')
tqdm.pandas()

# Replace this with your actual Gemini API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY') 
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# ---- Load Data ----
comments_df = pd.read_csv("UScomments.csv", engine='python', on_bad_lines='skip')
videos_df = pd.read_csv("USvideos.csv", engine='python', on_bad_lines='skip')

# ---- Merge and Preprocess ----
merged_df = pd.merge(comments_df, videos_df, on='video_id')
merged_df = merged_df[['video_id', 'comment_text', 'likes_x', 'title', 'tags']]
merged_df.rename(columns={'likes_x': 'comment_likes'}, inplace=True)
merged_df.dropna(subset=['comment_text'], inplace=True)
merged_df.drop_duplicates(subset=['comment_text'], inplace=True)

def clean_comment(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

merged_df['clean_comment'] = merged_df['comment_text'].apply(clean_comment)

# ---- Sentiment Analysis ----
analyzer = SentimentIntensityAnalyzer()
sentiment_scores = merged_df['clean_comment'].progress_apply(analyzer.polarity_scores)
sentiment_df = sentiment_scores.apply(pd.Series)
merged_df = pd.concat([merged_df, sentiment_df], axis=1)

# ---- Extract Keywords ----
stop_words = set(stopwords.words('english'))

def extract_keywords(text):
    words = text.split()
    return [word for word in words if word not in stop_words and len(word) > 2]

merged_df['keywords'] = merged_df['clean_comment'].apply(extract_keywords)

# ---- Group Keywords per Video ----
video_keywords = merged_df.groupby('video_id')['keywords'].sum()
video_keywords_top = video_keywords.apply(lambda words: [w for w, _ in Counter(words).most_common(5)])
video_keywords_df = video_keywords_top.reset_index()
video_keywords_df.columns = ['video_id', 'top_keywords']

# ---- Sentiment Summary per Video ----
video_sentiment = merged_df.groupby('video_id')[['compound', 'pos', 'neu', 'neg']].mean()
sample_comments = merged_df.groupby('video_id')['clean_comment'].first().reset_index()
sample_comments.rename(columns={'clean_comment': 'sample_comments'}, inplace=True)

summary_df = video_sentiment.merge(video_keywords_df, on='video_id')
summary_df = summary_df.merge(sample_comments, on='video_id')
summary_df = summary_df.merge(videos_df[['video_id', 'title']], on='video_id')

# ---- Prompt Creation ----
def sentiment_label(score):
    if score >= 0.3:
        return "positive"
    elif score <= -0.3:
        return "negative"
    else:
        return "neutral"
    
# ---- Build Simple Rule-Based Summary ----
def build_summary(row):
    label = sentiment_label(row['compound'])
    keywords_list = row['top_keywords']
    summary = f"Overall, people are *{label}* about this video."
    
    if keywords_list:
        summary += f" Top trending topics in the comments include: {', '.join(keywords_list)}."
    
    return summary

summary_df['summary'] = summary_df.apply(build_summary, axis=1)

def build_prompt(row):
    label = sentiment_label(row['compound'])
    keywords = ", ".join(row['top_keywords'])
    return (
        f"Video Title: {row['title']}\n"
        f"Viewer Sentiment: {label} "
        f"(compound: {row['compound']:.2f}, pos: {row['pos']:.2f}, neu: {row['neu']:.2f}, neg: {row['neg']:.2f})\n"
        f"Top Keywords: {keywords}\n"
        f"Sample Comment: {row['sample_comments']}\n\n"
        "Write a detailed, insightful summary of viewer opinion for this video. "
        "Incorporate the sentiment and top keywords, and avoid repeating phrases or points. "
        "Highlight any notable trends, concerns, or praise from viewers. "
        "Do not repeat information, and make the summary engaging and informative."
    )


summary_df['prompt'] = summary_df.apply(build_prompt, axis=1)

# ---- Gemini Response Function ----
def generate_gemini_summary(prompt):
    try:
        response = gemini_model.generate_content(prompt, generation_config={"max_output_tokens": 150})
        return response.text
    except Exception as e:
        print(f"Error generating Gemini summary: {e}")
        return "Error"

# ---- Generate Summaries ----
summary_df['gemini_response'] = summary_df['prompt'].head(10).apply(generate_gemini_summary)

# ---- Output Result ----
output = summary_df[['title', 'summary', 'gemini_response']].head(10)
output.to_csv("gemini_video_summary.csv", index=False)
print(output)
