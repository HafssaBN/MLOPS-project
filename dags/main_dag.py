# main_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import timedelta, datetime
import pandas as pd
import os
import sys
import logging

# Add the directory containing your custom modules to sys.path
# Adjust the path if your custom scripts are located elsewhere
sys.path.insert(0, '/opt/airflow/dags/')

# Import custom modules after updating sys.path
from scrapping import get_channel_stats, get_video_ids, get_video_details, get_comments_in_videos
from preprocessing import preprocess_video_data
from data_analysis import analyze_video_performance

# Configure logging at the top level to capture logs from all tasks
logging.basicConfig(
    filename='/opt/airflow/data/data_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Ensure the required directories exist
os.makedirs('/opt/airflow/data', exist_ok=True)
os.makedirs('/opt/airflow/data/plots', exist_ok=True)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # Ensures that each run is independent
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

api_key = 'AIzaSyAxs7uzwcGOqJftvR_p9gqZ2nT6KmwtKj4'

# Initialize the DAG
with DAG(
    'video_data_processing_dag',
    default_args=default_args,
    description='A DAG for scraping, preprocessing, and analyzing video data',
    catchup=False,
    schedule_interval=None,  # Set to a cron expression if you want scheduled runs
    start_date=datetime(2023, 1, 1),
    tags=['youtube', 'data_processing'],
) as dag:

    # Task 1: Print Start
    def print_start():
        message = "The DAG has started."
        print(message)
        logging.info(message)
        return "DAG Start"

    print_start_task = PythonOperator(
        task_id='print_start',
        python_callable=print_start,
    )

    # Task 2: Fetch Channel Statistics
    def fetch_channel_stats():
        try:
            from googleapiclient.discovery import build

            youtube = build('youtube', 'v3', developerKey=api_key)
            channel_ids = ['UCoOae5nYA7VqaXzerajD0lg']
            channel_stats = get_channel_stats(youtube, channel_ids)

            df = pd.DataFrame(channel_stats)
            df.to_csv('/opt/airflow/data/channel_stats.csv', index=False)
            logging.info("Channel statistics fetched and saved.")
        except Exception as e:
            logging.error(f"Error in fetch_channel_stats: {e}")
            raise

    fetch_channel_stats_task = PythonOperator(
        task_id='fetch_channel_stats',
        python_callable=fetch_channel_stats,
    )

    # Task 3: Fetch Video IDs
    def fetch_video_ids():
        try:
            from googleapiclient.discovery import build

            youtube = build('youtube', 'v3', developerKey=api_key)
            playlist_id = "UUoOae5nYA7VqaXzerajD0lg"
            video_ids = get_video_ids(youtube, playlist_id)

            df = pd.DataFrame(video_ids, columns=['video_id'])
            df.to_csv('/opt/airflow/data/video_ids.csv', index=False)
            logging.info("Video IDs fetched and saved.")
        except Exception as e:
            logging.error(f"Error in fetch_video_ids: {e}")
            raise

    fetch_video_ids_task = PythonOperator(
        task_id='fetch_video_ids',
        python_callable=fetch_video_ids,
    )

    # Task 4: Fetch Video Details
    def fetch_video_details():
        try:
            from googleapiclient.discovery import build

            youtube = build('youtube', 'v3', developerKey=api_key)
            video_ids = pd.read_csv('/opt/airflow/data/video_ids.csv')['video_id'].tolist()
            video_details = get_video_details(youtube, video_ids)

            df = pd.DataFrame(video_details)
            df.to_csv('/opt/airflow/data/video_details_raw.csv', index=False)
            logging.info("Video details fetched and saved.")
        except FileNotFoundError:
            logging.error("The file 'video_ids.csv' was not found in '/opt/airflow/data/'.")
            raise
        except Exception as e:
            logging.error(f"Error in fetch_video_details: {e}")
            raise

    fetch_video_details_task = PythonOperator(
        task_id='fetch_video_details',
        python_callable=fetch_video_details,
    )

    # Task 5: Preprocess Video Data
    def preprocess_data():
        try:
            df_raw = pd.read_csv('/opt/airflow/data/video_details_raw.csv')
            logging.info(f"Loaded 'video_details_raw.csv' with {len(df_raw)} records.")

            df_preprocessed = preprocess_video_data(df_raw)
            df_preprocessed.to_csv('/opt/airflow/data/video_details.csv', index=False)
            logging.info(f"Data preprocessing completed and saved 'video_details.csv' with {len(df_preprocessed)} records.")

            # Check for missing values in required columns
            required_columns = ['viewCount', 'likeCount', 'commentCount', 'durationSecs', 'publishDayName', 'title']
            missing_values = df_preprocessed[required_columns].isnull().sum()
            logging.info(f"Missing values after preprocessing:\n{missing_values}")

            if df_preprocessed[required_columns].isnull().any().any():
                logging.warning("There are missing values in required columns after preprocessing.")
        except FileNotFoundError:
            logging.error("The file 'video_details_raw.csv' was not found in '/opt/airflow/data/'.")
            raise
        except pd.errors.EmptyDataError:
            logging.error("'video_details_raw.csv' is empty.")
            raise
        except Exception as e:
            logging.error(f"Error in preprocess_data: {e}")
            raise

    preprocess_data_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    # Task 6: Fetch Enhanced Comments
    def fetch_comments():
        try:
            from googleapiclient.discovery import build

            youtube = build('youtube', 'v3', developerKey=api_key)
            video_ids = pd.read_csv('/opt/airflow/data/video_ids.csv')['video_id'].tolist()
            
            # Use the enhanced function to get detailed comments
            enhanced_comments = get_comments_in_videos(youtube, video_ids, max_comments_per_video=30)
            
            if not enhanced_comments.empty:
                # Save the enhanced comments data
                enhanced_comments.to_csv('/opt/airflow/data/comments_detailed.csv', index=False)
                
                # Create a simplified version for backwards compatibility
                simplified_comments = enhanced_comments[['video_id', 'comment_text']].rename(columns={'comment_text': 'comments'})
                # Group comments by video_id for the old format
                grouped_comments = simplified_comments.groupby('video_id')['comments'].apply(list).reset_index()
                grouped_comments.to_csv('/opt/airflow/data/comments.csv', index=False)
                
                # Create summary statistics
                comment_stats = enhanced_comments.groupby('video_id').agg({
                    'comment_id': 'count',
                    'like_count': ['mean', 'sum', 'max'],
                    'comment_length': ['mean', 'median'],
                    'word_count': ['mean', 'median'],
                    'has_mentions': 'sum',
                    'has_hashtags': 'sum',
                    'has_urls': 'sum',
                    'total_reply_count': 'sum',
                    'is_reply': 'sum'
                }).round(2)
                
                # Flatten column names
                comment_stats.columns = ['_'.join(col).strip() for col in comment_stats.columns]
                comment_stats = comment_stats.reset_index()
                
                # Rename columns for clarity
                column_mapping = {
                    'comment_id_count': 'total_comments',
                    'like_count_mean': 'avg_likes_per_comment',
                    'like_count_sum': 'total_likes',
                    'like_count_max': 'max_likes_per_comment',
                    'comment_length_mean': 'avg_comment_length',
                    'comment_length_median': 'median_comment_length',
                    'word_count_mean': 'avg_word_count',
                    'word_count_median': 'median_word_count',
                    'has_mentions_sum': 'comments_with_mentions',
                    'has_hashtags_sum': 'comments_with_hashtags',
                    'has_urls_sum': 'comments_with_urls',
                    'total_reply_count_sum': 'total_replies',
                    'is_reply_sum': 'reply_comments_count'
                }
                comment_stats = comment_stats.rename(columns=column_mapping)
                comment_stats.to_csv('/opt/airflow/data/comment_statistics.csv', index=False)
                
                logging.info(f"Enhanced comments fetched and saved. Total comments: {len(enhanced_comments)}")
                logging.info(f"Columns in enhanced comments: {enhanced_comments.columns.tolist()}")
                logging.info(f"Comment statistics saved with {len(comment_stats)} video records.")
            else:
                logging.warning("No comments were fetched.")
                
        except FileNotFoundError:
            logging.error("The file 'video_ids.csv' was not found in '/opt/airflow/data/'.")
            raise
        except Exception as e:
            logging.error(f"Error in fetch_comments: {e}")
            raise

    fetch_comments_task = PythonOperator(
        task_id='fetch_comments',
        python_callable=fetch_comments,
    )

    # Task 7: Analyze Comment Patterns (New Task)
    def analyze_comment_patterns():
        try:
            # Load the detailed comments data
            detailed_comments = pd.read_csv('/opt/airflow/data/comments_detailed.csv')
            
            if detailed_comments.empty:
                logging.warning("No detailed comments data found for analysis.")
                return
            
            # Analyze comment patterns
            logging.info("Starting comment pattern analysis...")
            
            # Top commenters analysis
            top_commenters = detailed_comments.groupby('author_name').agg({
                'comment_id': 'count',
                'like_count': 'sum',
                'video_id': 'nunique'
            }).rename(columns={
                'comment_id': 'total_comments',
                'like_count': 'total_likes_received',
                'video_id': 'videos_commented_on'
            }).sort_values('total_comments', ascending=False).head(20)
            
            top_commenters.to_csv('/opt/airflow/data/top_commenters.csv')
            
            # Comment timing analysis
            detailed_comments['published_at'] = pd.to_datetime(detailed_comments['published_at'])
            detailed_comments['hour'] = detailed_comments['published_at'].dt.hour
            detailed_comments['day_of_week'] = detailed_comments['published_at'].dt.day_name()
            
            hourly_comments = detailed_comments.groupby('hour').size().reset_index(name='comment_count')
            daily_comments = detailed_comments.groupby('day_of_week').size().reset_index(name='comment_count')
            
            hourly_comments.to_csv('/opt/airflow/data/hourly_comment_patterns.csv', index=False)
            daily_comments.to_csv('/opt/airflow/data/daily_comment_patterns.csv', index=False)
            
            # Engagement analysis
            engagement_stats = detailed_comments.groupby('video_id').agg({
                'like_count': ['mean', 'sum', 'std'],
                'comment_length': ['mean', 'std'],
                'word_count': ['mean', 'std'],
                'total_reply_count': 'sum'
            }).round(2)
            
            engagement_stats.columns = ['_'.join(col).strip() for col in engagement_stats.columns]
            engagement_stats = engagement_stats.reset_index()
            engagement_stats.to_csv('/opt/airflow/data/comment_engagement_analysis.csv', index=False)
            
            logging.info("Comment pattern analysis completed successfully.")
            
        except FileNotFoundError:
            logging.error("The file 'comments_detailed.csv' was not found.")
        except Exception as e:
            logging.error(f"Error in analyze_comment_patterns: {e}")
            raise

    analyze_comment_patterns_task = PythonOperator(
        task_id='analyze_comment_patterns',
        python_callable=analyze_comment_patterns,
    )

    # Task 8: Analyze Performance
    def analyze_performance():
        try:
            df_video = pd.read_csv('/opt/airflow/data/video_details.csv')
            logging.info("Loaded 'video_details.csv' successfully.")
            logging.info(f"Columns in 'video_details.csv': {df_video.columns.tolist()}")
            logging.info(f"Number of records in 'video_details.csv': {len(df_video)}")

            # Check data types
            logging.info("Data types in 'video_details.csv':")
            logging.info(df_video.dtypes)

            # Check for any NaN values in required columns
            required_columns = ['viewCount', 'likeCount', 'commentCount', 'durationSecs', 'publishDayName', 'title']
            missing_values = df_video[required_columns].isnull().sum()
            logging.info(f"Missing values in required columns:\n{missing_values}")

            analyze_video_performance(df_video)
            logging.info("Video performance analysis completed and plots saved.")
        except FileNotFoundError:
            logging.error("The file 'video_details.csv' was not found in '/opt/airflow/data/'.")
            raise
        except pd.errors.EmptyDataError:
            logging.error("'video_details.csv' is empty.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred in 'analyze_performance': {e}")
            raise

    analyze_performance_task = PythonOperator(
        task_id='analyze_performance',
        python_callable=analyze_performance,
    )

    # Task 9: Generate Final Report (New Task)
    def generate_final_report():
        try:
            logging.info("Generating final data processing report...")
            
            # Load all the generated files and create a summary report
            files_to_check = [
                '/opt/airflow/data/channel_stats.csv',
                '/opt/airflow/data/video_details.csv',
                '/opt/airflow/data/comments_detailed.csv',
                '/opt/airflow/data/comment_statistics.csv',
                '/opt/airflow/data/top_commenters.csv'
            ]
            
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'files_generated': [],
                'data_summary': {}
            }
            
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    filename = os.path.basename(file_path)
                    report['files_generated'].append(filename)
                    report['data_summary'][filename] = {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'size_mb': round(os.path.getsize(file_path) / (1024*1024), 2)
                    }
            
            # Save report as JSON and CSV
            import json
            with open('/opt/airflow/data/processing_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Create a simple CSV summary
            summary_data = []
            for filename, stats in report['data_summary'].items():
                summary_data.append({
                    'filename': filename,
                    'rows': stats['rows'],
                    'columns': stats['columns'],
                    'size_mb': stats['size_mb']
                })
            
            pd.DataFrame(summary_data).to_csv('/opt/airflow/data/processing_summary.csv', index=False)
            
            logging.info(f"Final report generated. Processed {len(report['files_generated'])} files successfully.")
            
        except Exception as e:
            logging.error(f"Error in generate_final_report: {e}")
            raise

    generate_final_report_task = PythonOperator(
        task_id='generate_final_report',
        python_callable=generate_final_report,
    )

    # Define Task Dependencies
    print_start_task >> fetch_channel_stats_task >> fetch_video_ids_task
    fetch_video_ids_task >> [fetch_video_details_task, fetch_comments_task]
    fetch_video_details_task >> preprocess_data_task
    fetch_comments_task >> analyze_comment_patterns_task
    preprocess_data_task >> analyze_performance_task
    [analyze_comment_patterns_task, analyze_performance_task] >> generate_final_report_task