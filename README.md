YouTube Video Sentiment Summarization (GenAI)

This project analyzes YouTube video comments and generates both rule-based and AI-powered sentiment summaries for each video. It uses Pandas, NLTK, VADER, Google Gemini API, and Tkinter for an interactive desktop interface.

Features:
Loads and cleans YouTube comments and video data,
Performs sentiment analysis using VADER,
Extracts trending keywords from comments,
Generates rule-based and Gemini AI summaries for each video,
Simple GUI to select a video and view summaries.

Requirements:
Python 3.8+,
pandas, 
numpy, 
nltk, 
vaderSentiment, 
tqdm, 
python-dotenv, 
google-generativeai, 
tkinter
Gemini API key (add to .env as GOOGLE_API_KEY),
NLTK stopwords (nltk.download('stopwords')),
UScomments.csv and USvideos.csv files in the project folder,

How to Run:
Install dependencies:
--pip install -r requirements.txt

Download NLTK stopwords:
--import nltk
--nltk.download('stopwords')

Add your Gemini API key to a .env file:
--GOOGLE_API_KEY=your_key_here

Place UScomments.csv and USvideos.csv in the project directory.

Run the script:
--python main.py

Use the GUI to select a video and view its sentiment summaries.

Output
Summaries for the top 10 videos are saved to gemini_video_summary.csv
