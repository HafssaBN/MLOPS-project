import pandas as pd
import isodate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_video_data(df):
    # Define expected columns, including 'title' which was handled later in the original code
    expected_columns = [
        'viewCount', 
        'likeCount', 
        'favouriteCount', 
        'commentCount', 
        'publishedAt', 
        'duration', 
        'tags', 
        'title'  # Added 'title' to handle missing titles
    ]
    
    # Check and ensure required columns exist
    for col in expected_columns:
        if col not in df.columns:
            if col == 'title':
                # For 'title', we'll fill missing values with 'No Title' later
                df[col] = None
            else:
                df[col] = None
            logging.warning(f"Missing column: {col}. Adding column with default values.")

    # Safely convert numeric columns to numeric
    numeric_cols = ['viewCount', 'likeCount', 'favouriteCount', 'commentCount']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Non-numeric values converted to NaN
        if df[col].isnull().any():
            logging.warning(f"Some values in '{col}' could not be converted to numeric and were set to NaN.")

    # Convert 'publishedAt' to datetime and extract day name
    try:
        df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
        df['publishDayName'] = df['publishedAt'].dt.strftime('%A')
    except Exception as e:
        logging.error(f"Error parsing 'publishedAt' column: {e}")
        df['publishDayName'] = None

    # Safely parse 'duration' and convert to seconds
    def safe_parse_duration(duration):
        try:
            return isodate.parse_duration(duration).total_seconds()
        except Exception:
            return None

    df['durationSecs'] = df['duration'].apply(safe_parse_duration)
    if df['durationSecs'].isnull().any():
        logging.warning("Some durations could not be parsed and were set to None.")

    # Safely count the number of tags
    def safe_count_tags(tags):
        try:
            return len(tags) if isinstance(tags, list) else 0
        except Exception:
            return 0

    df['tagCount'] = df['tags'].apply(safe_count_tags)
    if (df['tags'].isnull()).any():
        logging.warning("Some 'tags' entries were missing or not lists and were set to 0.")

    # Fill missing titles with a default value
    if 'title' in df.columns:
        missing_titles = df['title'].isnull().sum()
        if missing_titles > 0:
            logging.warning(f"{missing_titles} titles were missing and have been filled with 'No Title'.")
        df['title'] = df['title'].fillna('No Title')
    else:
        # This case should not occur since 'title' is added to expected_columns
        df['title'] = 'No Title'
        logging.warning("'title' column was missing and has been created with default values.")

    # Log data types after preprocessing
    logging.info("Data types after preprocessing:")
    logging.info(df.dtypes)

    # Log the number of records and missing values
    logging.info(f"Number of records after preprocessing: {len(df)}")
    logging.info(f"Missing values in each column:\n{df.isnull().sum()}")

    logging.info("Preprocessing completed successfully.")
    
    return df
