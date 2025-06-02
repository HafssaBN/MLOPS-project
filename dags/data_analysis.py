import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wordcloud import WordCloud
from nltk.corpus import stopwords
import seaborn as sns
import pandas as pd
import os
import re
import logging
import nltk

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_plot(fig, filename):
    """
    Save the plot to the specified directory.
    """
    try:
        output_dir = '/opt/airflow/data/plots'
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight')
        logging.info(f"Plot saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving plot {filename}: {e}")
    finally:
        plt.close(fig)

def analyze_video_performance(video_df):
    """
    Analyze video performance: Generate plots for best videos, upload schedule, and word cloud for titles.
    """
    try:
        # Validate required columns
        required_cols = ['viewCount', 'likeCount', 'commentCount', 'durationSecs', 'publishDayName', 'title']
        for col in required_cols:
            if col not in video_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Drop rows with missing numeric data
        numeric_cols = ['viewCount', 'likeCount', 'commentCount', 'durationSecs']
        video_df = video_df.dropna(subset=numeric_cols)
        if video_df.empty:
            logging.error("No valid rows found after dropping invalid data.")
            return

        # Ensure numeric columns are properly formatted
        for col in numeric_cols:
            video_df[col] = pd.to_numeric(video_df[col], errors='coerce').fillna(0).astype(int)

        # Clean and prepare titles for visualization
        video_df['title'] = video_df['title'].fillna('No Title')
        video_df['title_plot'] = video_df['title'].apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x))

        # Top 9 Best Performing Videos
        top_videos = video_df.sort_values('viewCount', ascending=False).head(9)
        if not top_videos.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='title_plot', y='viewCount', data=top_videos, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
            plt.title("Top 9 Best Performing Videos")
            save_plot(fig, 'top_9_best_performing_videos.png')
        else:
            logging.warning("No valid data available for Top 9 Best Performing Videos.")

        # Upload Schedule by Day
        day_counts = video_df['publishDayName'].value_counts()
        if not day_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax)
            plt.title("Upload Schedule by Day")
            save_plot(fig, 'upload_schedule_by_day.png')
        else:
            logging.warning("No valid data available for Upload Schedule.")

        # Word Cloud for Titles
        stop_words = set(stopwords.words('english'))
        text = ' '.join(video_df['title_plot'])
        clean_text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
        if clean_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean_text)
            fig = plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Word Cloud for Video Titles")
            save_plot(fig, 'wordcloud_titles.png')
        else:
            logging.warning("No valid text available for Word Cloud generation.")

        # Video Duration Analysis
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(video_df['durationSecs'], bins=30, kde=True, ax=ax)
        plt.title("Video Duration Distribution")
        plt.xlabel("Duration (Seconds)")
        save_plot(fig, 'video_duration_distribution.png')

        # Views vs. Likes and Comments
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.scatterplot(data=video_df, x='likeCount', y='viewCount', ax=axes[0])
        axes[0].set_title("Views vs. Likes")
        axes[0].set_xlabel("Likes")
        axes[0].set_ylabel("Views")

        sns.scatterplot(data=video_df, x='commentCount', y='viewCount', ax=axes[1])
        axes[1].set_title("Views vs. Comments")
        axes[1].set_xlabel("Comments")
        axes[1].set_ylabel("Views")

        save_plot(fig, 'views_vs_likes_and_comments.png')

        logging.info("Video performance analysis completed successfully.")

    except Exception as e:
        logging.error(f"Error in analyze_video_performance: {e}")

