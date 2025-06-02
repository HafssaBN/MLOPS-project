from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime  # Direct import for clarity
import os
import logging

logging.basicConfig(
    filename='/opt/airflow/data_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Function to get channel stats using YouTube API
def get_channel_stats(youtube, channel_ids):
    """
    Scrape channel statistics like subscribers, views, and videos.
    
    Parameters:
    youtube: build object of the YouTube API client
    channel_ids: list of channel IDs to scrape
    
    Returns:
    A DataFrame containing channel stats for each channel
    """
    all_data = []
   
    # Making API request to get channel details
    request = youtube.channels().list(
        part="snippet,contentDetails,statistics",
        id=','.join(channel_ids)
    )
    response = request.execute()

    # Loop through the response and extract useful data
    for item in response['items']:
        data = {
            'channelName': item['snippet']['title'],
            'subscribers': item['statistics']['subscriberCount'],
            'views': item['statistics']['viewCount'],
            'totalVideos': item['statistics']['videoCount'],
            'playlistId': item['contentDetails']['relatedPlaylists']['uploads']
        }
        all_data.append(data)

    # Return the data as a DataFrame
    return pd.DataFrame(all_data)

# Function to get video IDs from a playlist
def get_video_ids(youtube, playlist_id):
    video_ids = []

    request = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=playlist_id,
        maxResults=50
    )
    response = request.execute()

    for item in response['items']:
        video_ids.append(item['contentDetails']['videoId'])

    next_page_token = response.get('nextPageToken')
    while next_page_token is not None:
        request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            video_ids.append(item['contentDetails']['videoId'])

        next_page_token = response.get('nextPageToken')

    return video_ids

# Function to get video details
def get_video_details(youtube, video_ids):
    all_video_info = []

    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids[i:i+50])
        )
        response = request.execute()

        for video in response['items']:
            stats_to_keep = {
                'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
                'statistics': ['viewCount', 'likeCount', 'favouriteCount', 'commentCount'],
                'contentDetails': ['duration', 'definition', 'caption']
            }
            video_info = {'video_id': video['id']}

            for k in stats_to_keep.keys():
                for v in stats_to_keep[k]:
                    try:
                        video_info[v] = video[k][v]
                    except KeyError:
                        video_info[v] = None

            all_video_info.append(video_info)

    return pd.DataFrame(all_video_info)

# Enhanced function to get comments from videos with more features
def get_comments_in_videos(youtube, video_ids, max_comments_per_video=20):
    """
    Enhanced function to get detailed comment information from videos
    
    Parameters:
    youtube: YouTube API client
    video_ids: List of video IDs
    max_comments_per_video: Maximum number of comments to fetch per video
    
    Returns:
    DataFrame with detailed comment information
    """
    all_comments = []

    for video_id in video_ids:
        try:
            # Get video title for context (optional)
            video_request = youtube.videos().list(
                part="snippet",
                id=video_id
            )
            video_response = video_request.execute()
            video_title = video_response['items'][0]['snippet']['title'] if video_response['items'] else 'Unknown'
            
            # Get comments
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(max_comments_per_video, 100),  # API limit is 100
                textFormat="plainText",
                order="relevance"
            )
            response = request.execute()

            comment_count = 0
            
            for comment_thread in response.get('items', []):
                if comment_count >= max_comments_per_video:
                    break
                    
                # Extract top-level comment details
                top_comment = comment_thread['snippet']['topLevelComment']['snippet']
                
                # Calculate comment length and word count
                comment_text = top_comment['textOriginal']
                comment_length = len(comment_text)
                word_count = len(comment_text.split())
                
                # Check if comment contains certain elements
                has_mentions = '@' in comment_text
                has_hashtags = '#' in comment_text
                has_urls = 'http' in comment_text.lower() or 'www.' in comment_text.lower()
                
                comment_data = {
                    'video_id': video_id,
                    'video_title': video_title,
                    'comment_id': comment_thread['snippet']['topLevelComment']['id'],
                    'comment_text': comment_text,
                    'comment_text_display': top_comment.get('textDisplay', ''),
                    'comment_length': comment_length,
                    'word_count': word_count,
                    'has_mentions': has_mentions,
                    'has_hashtags': has_hashtags,
                    'has_urls': has_urls,
                    'author_name': top_comment['authorDisplayName'],
                    'author_channel_id': top_comment.get('authorChannelId', {}).get('value', ''),
                    'author_profile_image': top_comment.get('authorProfileImageUrl', ''),
                    'like_count': int(top_comment.get('likeCount', 0)),
                    'published_at': top_comment['publishedAt'],
                    'updated_at': top_comment.get('updatedAt', top_comment['publishedAt']),
                    'is_reply': False,
                    'parent_comment_id': None,
                    'total_reply_count': int(comment_thread['snippet'].get('totalReplyCount', 0)),
                    'can_rate': comment_thread['snippet']['topLevelComment']['snippet'].get('canRate', False),
                    'viewer_rating': comment_thread['snippet']['topLevelComment']['snippet'].get('viewerRating', 'none'),
                    'moderation_status': top_comment.get('moderationStatus', 'published'),
                    'comment_type': 'top_level'
                }
                
                all_comments.append(comment_data)
                comment_count += 1
                
                # Extract replies if they exist and we haven't reached the limit
                if 'replies' in comment_thread and comment_count < max_comments_per_video:
                    for reply in comment_thread['replies']['comments']:
                        if comment_count >= max_comments_per_video:
                            break
                            
                        reply_snippet = reply['snippet']
                        reply_text = reply_snippet['textOriginal']
                        reply_length = len(reply_text)
                        reply_word_count = len(reply_text.split())
                        
                        reply_has_mentions = '@' in reply_text
                        reply_has_hashtags = '#' in reply_text
                        reply_has_urls = 'http' in reply_text.lower() or 'www.' in reply_text.lower()
                        
                        reply_data = {
                            'video_id': video_id,
                            'video_title': video_title,
                            'comment_id': reply['id'],
                            'comment_text': reply_text,
                            'comment_text_display': reply_snippet.get('textDisplay', ''),
                            'comment_length': reply_length,
                            'word_count': reply_word_count,
                            'has_mentions': reply_has_mentions,
                            'has_hashtags': reply_has_hashtags,
                            'has_urls': reply_has_urls,
                            'author_name': reply_snippet['authorDisplayName'],
                            'author_channel_id': reply_snippet.get('authorChannelId', {}).get('value', ''),
                            'author_profile_image': reply_snippet.get('authorProfileImageUrl', ''),
                            'like_count': int(reply_snippet.get('likeCount', 0)),
                            'published_at': reply_snippet['publishedAt'],
                            'updated_at': reply_snippet.get('updatedAt', reply_snippet['publishedAt']),
                            'is_reply': True,
                            'parent_comment_id': comment_thread['snippet']['topLevelComment']['id'],
                            'total_reply_count': 0,  # Replies don't have reply counts
                            'can_rate': reply_snippet.get('canRate', False),
                            'viewer_rating': reply_snippet.get('viewerRating', 'none'),
                            'moderation_status': reply_snippet.get('moderationStatus', 'published'),
                            'comment_type': 'reply'
                        }
                        
                        all_comments.append(reply_data)
                        comment_count += 1

            logging.info(f"Extracted {comment_count} comments from video {video_id}")

        except Exception as e:
            # Log the error for debugging purposes
            error_msg = f'Could not get comments for video {video_id}: {e}'
            print(error_msg)
            logging.error(error_msg)

    return pd.DataFrame(all_comments)

# Legacy function for backwards compatibility (simple version)
def get_simple_comments_in_videos(youtube, video_ids):
    """
    Simple function to get basic comments (for backwards compatibility)
    """
    all_comments = []

    for video_id in video_ids:
        try:
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=10,  # Limit to first 10 comments
                textFormat="plainText"
            )
            response = request.execute()

            comments_in_video = [
                comment['snippet']['topLevelComment']['snippet']['textOriginal']
                for comment in response.get('items', [])
            ]
            comments_in_video_info = {'video_id': video_id, 'comments': comments_in_video}

            all_comments.append(comments_in_video_info)

        except Exception as e:
            # Log the error for debugging purposes
            print(f'Could not get comments for video {video_id}: {e}')

    return pd.DataFrame(all_comments)

# Function to save the scraped data to a CSV file with a timestamp
def save_scraped_data(df, prefix="scraped_data"):
    # Use the directory we mounted as a volume
    output_dir = "Youtube-Web-scrapping-and-data-pipeline-in-Airflow/data"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")

# Main function to orchestrate the entire process
def main(youtube, channel_ids, playlist_id):
    # Get channel stats
    channel_stats = get_channel_stats(youtube, channel_ids)
    print("Channel Stats:", channel_stats)

    # Get video IDs from playlist
    video_ids = get_video_ids(youtube, playlist_id)
    print("Video IDs:", len(video_ids), video_ids[:5])

    # Get video details
    video_details = get_video_details(youtube, video_ids)
    print("Video Details:", video_details)

    # Get enhanced comments from videos
    comments = get_comments_in_videos(youtube, video_ids)
    print("Enhanced Comments:", comments)

    # Save data to CSV
    save_scraped_data(channel_stats, "channel_stats")
    save_scraped_data(video_details, "video_details")
    save_scraped_data(comments, "enhanced_comments")