import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df= pd.read_csv('data/comments_detailed.csv')
import pandas as pd
import string
import re
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import time
import optuna
import pickle
import json

# Download stopwords once
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Enhanced cleaning function
stop_words = set(stopwords.words('english'))

def clean_doc(doc):
    """Enhanced text cleaning with more preprocessing steps"""
    if pd.isna(doc):
        return []

    doc = str(doc).lower()  # Convert to lowercase

    # Remove URLs, emails, and mentions
    doc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', doc)
    doc = re.sub(r'\S+@\S+', '', doc)  # Remove emails
    doc = re.sub(r'@\w+', '', doc)     # Remove mentions

    # Tokenize
    tokens = doc.split()

    # Remove punctuation
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]

    # Filter tokens
    tokens = [w for w in tokens if w.isalpha()]  # Only alphabetic
    tokens = [w for w in tokens if w not in stop_words]  # Remove stopwords
    tokens = [w for w in tokens if len(w) > 2]  # Minimum length 3

    return tokens

def detect_toxic_bot_patterns(df):
    """
    Enhanced function to detect toxic and bot comment patterns
    Returns a DataFrame with additional features for better classification
    """

    # Toxic keywords (expand this based on your domain)
    toxic_keywords = {
        'hate_speech': ['hate', 'stupid', 'idiot', 'moron', 'dumb', 'trash', 'garbage', 'worthless'],
        'profanity': ['damn', 'hell', 'crap', 'suck', 'sucks', 'awful', 'terrible', 'worst'],
        'spam_words': ['subscribe', 'follow', 'like', 'check', 'visit', 'click', 'free', 'win', 'prize'],
        'bot_phrases': ['first', 'early', 'notification squad', 'who else', 'anyone else', 'copy paste']
    }

    # Bot behavior patterns
    def extract_features(row):
        text = str(row['comment_text']).lower()

        features = {}

        # Basic text features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)

        # Repetition patterns (bot-like behavior)
        words = text.split()
        features['repeated_words'] = len(words) - len(set(words)) if words else 0
        features['repeated_chars'] = max([len(list(group)) for char, group in
                                        __import__('itertools').groupby(text)] + [0])

        # Toxic keyword counts
        for category, keywords in toxic_keywords.items():
            features[f'{category}_count'] = sum(1 for keyword in keywords if keyword in text)

        # Bot-specific patterns
        features['contains_subscribe'] = 1 if any(word in text for word in ['subscribe', 'sub']) else 0
        features['contains_like_request'] = 1 if any(phrase in text for phrase in ['like if', 'thumbs up']) else 0
        features['contains_generic_praise'] = 1 if any(phrase in text for phrase in ['great video', 'nice video', 'good job']) else 0
        features['is_very_short'] = 1 if len(words) <= 3 else 0
        features['is_very_long'] = 1 if len(words) > 100 else 0

        # Time-based features (if available)
        if 'published_at' in row and pd.notna(row['published_at']):
            # Add time-based features here if needed
            pass

        # Engagement features
        features['like_count'] = row.get('like_count', 0)
        features['has_replies'] = 1 if row.get('total_reply_count', 0) > 0 else 0
        features['is_reply'] = 1 if row.get('is_reply', False) else 0

        return features

    # Extract features for all comments
    print("Extracting features for toxic/bot detection...")
    feature_data = []
    for idx, row in df.iterrows():
        features = extract_features(row)
        features['original_index'] = idx
        feature_data.append(features)

    feature_df = pd.DataFrame(feature_data)

    return feature_df

def create_toxic_bot_labels(df, feature_df, method='heuristic'):
    """
    Create labels for toxic/bot comments using different methods

    Parameters:
    - method: 'heuristic' (rule-based), 'unsupervised' (clustering), or 'manual' (if you have labeled data)
    """

    if method == 'heuristic':
        # Rule-based labeling - this is a starting point, adjust based on your data
        labels = []

        for idx, (_, row) in enumerate(df.iterrows()):
            features = feature_df.iloc[idx]

            # Toxic comment indicators
            toxic_score = 0

            # High toxic keyword usage
            toxic_score += features['hate_speech_count'] * 3
            toxic_score += features['profanity_count'] * 2

            # Excessive caps or punctuation (aggressive tone)
            if features['caps_ratio'] > 0.3:
                toxic_score += 2
            if features['exclamation_count'] > 3:
                toxic_score += 1

            # Bot comment indicators
            bot_score = 0

            # Spam-like content
            bot_score += features['spam_words_count'] * 2
            bot_score += features['bot_phrases_count'] * 2

            # Generic/repetitive content
            if features['contains_subscribe']:
                bot_score += 2
            if features['contains_like_request']:
                bot_score += 2
            if features['contains_generic_praise']:
                bot_score += 1

            # Very short, generic comments
            if features['is_very_short'] and features['like_count'] == 0:
                bot_score += 1

            # Excessive repetition
            if features['repeated_words'] > 2:
                bot_score += 2
            if features['repeated_chars'] > 3:
                bot_score += 1

            # Final labeling (1 = toxic/bot, 0 = normal)
            total_score = toxic_score + bot_score

            # You can adjust these thresholds based on your data
            if total_score >= 3:
                labels.append(1)  # Toxic/Bot
            elif toxic_score >= 2 or bot_score >= 3:
                labels.append(1)  # Toxic/Bot
            else:
                labels.append(0)  # Normal

        return np.array(labels)

    elif method == 'unsupervised':
        # Use clustering to identify potential toxic/bot comments
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        # Select relevant features for clustering
        cluster_features = [
            'hate_speech_count', 'profanity_count', 'spam_words_count',
            'caps_ratio', 'repeated_words', 'is_very_short'
        ]

        X_cluster = feature_df[cluster_features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)

        # Perform clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        # Determine which cluster represents toxic/bot comments
        cluster_means = {}
        for cluster in [0, 1]:
            mask = clusters == cluster
            cluster_means[cluster] = X_cluster[mask].mean()

        # Assume cluster with higher toxic indicators is the toxic/bot cluster
        toxic_cluster = max(cluster_means.keys(),
                          key=lambda x: cluster_means[x]['hate_speech_count'] +
                                      cluster_means[x]['spam_words_count'])

        labels = (clusters == toxic_cluster).astype(int)
        return labels

    else:
        raise ValueError("Method must be 'heuristic' or 'unsupervised'")

def preprocess_data_toxic_bot(df, test_size=0.2, max_features=5000, labeling_method='heuristic'):
    """Preprocess data specifically for toxic/bot detection"""

    print("Preprocessing data for toxic/bot comment detection...")

    # Clean text data
    print("Cleaning text data...")
    df['clean_tokens'] = df['comment_text'].apply(clean_doc)
    df['clean_text'] = df['clean_tokens'].apply(lambda x: ' '.join(x))

    # Extract features for toxic/bot detection
    feature_df = detect_toxic_bot_patterns(df)

    # Create labels using specified method
    print(f"Creating labels using {labeling_method} method...")
    labels = create_toxic_bot_labels(df, feature_df, method=labeling_method)

    # Add labels to dataframe
    df['toxic_bot_label'] = labels

    # Remove empty comments
    valid_mask = (df['clean_text'].str.len() > 0) & (df['comment_text'].notna())
    df = df[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask]

    print(f"Label distribution: {Counter(labels)}")
    print(f"Percentage of toxic/bot comments: {(labels.sum() / len(labels) * 100):.2f}%")

    # Check for class imbalance
    class_counts = Counter(labels)
    if len(class_counts) < 2:
        print("Warning: Only one class found! Adjusting labeling criteria...")
        # Fallback: use a more lenient threshold
        feature_df = detect_toxic_bot_patterns(df)
        labels = create_toxic_bot_labels(df, feature_df, method='heuristic')
        # Make labeling more sensitive
        labels = np.where(
            (feature_df['hate_speech_count'] > 0) |
            (feature_df['spam_words_count'] > 1) |
            (feature_df['caps_ratio'] > 0.4), 1, 0
        )
        class_counts = Counter(labels)
        print(f"Adjusted label distribution: {class_counts}")

    # Calculate class weights for imbalanced datasets
    total_samples = len(labels)
    class_weights = {}
    for class_label in class_counts.keys():
        class_weights[class_label] = total_samples / (len(class_counts) * class_counts[class_label])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], labels,
        test_size=test_size,
        random_state=42,
        stratify=labels
    )

    # Tokenize with limited vocabulary
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)

    X_train_matrix = tokenizer.texts_to_matrix(X_train, mode='tfidf')
    X_test_matrix = tokenizer.texts_to_matrix(X_test, mode='tfidf')

    return X_train_matrix, X_test_matrix, y_train, y_test, tokenizer, class_weights, df

# [Keep all the neural network classes and training functions from your original code]
class ImprovedMLP(nn.Module):
    """Improved MLP with better regularization"""
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(ImprovedMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),

            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        return self.network(x)

class ImprovedDeepNN(nn.Module):
    """Improved deeper network with skip connections"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.4):
        super(ImprovedDeepNN, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (0.8 ** i))
            ))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class ImprovedResNet(nn.Module):
    """Improved ResNet with better skip connections"""
    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, dropout_rate=0.3):
        super(ImprovedResNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.residual_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.residual_blocks.append(self._make_residual_block(hidden_dim, dropout_rate))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _make_residual_block(self, dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = torch.relu(x)

        return self.output_layer(x)

# [Include all training and evaluation functions from original code]
def train_model_fixed(model, train_loader, val_loader, test_loader, criterion, optimizer,
                     scheduler=None, epochs=50, patience=10, device='cuda', class_weights=None):
    """Improved training with proper loss handling and test evaluation"""

    history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'train_acc': [],
        'val_acc': [],
        'test_acc': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None

    if class_weights is not None:
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()

            loss = criterion(outputs, batch_y.squeeze())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predicted == batch_y.squeeze()).sum().item()
            train_total += batch_y.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()

                loss = criterion(outputs, batch_y.squeeze())
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predicted == batch_y.squeeze()).sum().item()
                val_total += batch_y.size(0)

            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()

                loss = criterion(outputs, batch_y.squeeze())
                test_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                test_correct += (predicted == batch_y.squeeze()).sum().item()
                test_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_test_loss = test_loss / len(test_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        test_accuracy = test_correct / test_total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['test_acc'].append(test_accuracy)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        if val_accuracy > best_val_acc:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f} (Acc: {train_accuracy:.4f}) | '
                  f'Val Loss: {avg_val_loss:.4f} (Acc: {val_accuracy:.4f}) | '
                  f'Test Acc: {test_accuracy:.4f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1} (patience: {patience})')
            break

    if best_model_state:
        model.load_state_dict(best_model_state)

    history['best_val_loss'] = best_val_loss
    history['best_val_acc'] = best_val_acc
    history['total_epochs'] = epoch + 1

    return history

def evaluate_model_comprehensive(model, test_loader, device='cuda'):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()

            loss = criterion(outputs, batch_y.squeeze())
            total_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    avg_loss = total_loss / len(test_loader)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_loss': avg_loss,
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'labels': all_labels
    }

# Main function for toxic/bot detection
def main_toxic_bot_detection(df, n_trials=50, timeout=3600, labeling_method='heuristic'):
    """Main function for toxic/bot comment detection"""

    print("Starting toxic/bot comment detection with Optuna optimization...")

    # Preprocess data specifically for toxic/bot detection
    X_train_full, X_test, y_train_full, y_test, tokenizer, class_weights, processed_df = preprocess_data_toxic_bot(
        df, labeling_method=labeling_method
    )

    # Split training data into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Class weights: {class_weights}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = X_train.shape[1]

    # Create Optuna study for hyperparameter optimization
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )

    # Create objective function
    def objective(trial):
        try:
            architecture = trial.suggest_categorical('architecture', ['ImprovedMLP', 'ImprovedDeepNN', 'ImprovedResNet'])
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

            if architecture == 'ImprovedMLP':
                model = ImprovedMLP(input_dim, hidden_dim, dropout_rate)
            elif architecture == 'ImprovedDeepNN':
                num_layers = trial.suggest_int('num_layers', 2, 3)
                hidden_dims = [hidden_dim // (2**i) for i in range(num_layers)]
                hidden_dims = [max(32, dim) for dim in hidden_dims]
                model = ImprovedDeepNN(input_dim, hidden_dims, dropout_rate)
            else:
                num_blocks = trial.suggest_int('num_blocks', 1, 3)
                model = ImprovedResNet(input_dim, hidden_dim, num_blocks, dropout_rate)

            model = model.to(device)

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32)
            )
            test_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=3, factor=0.7, verbose=False
            )

            criterion = nn.BCEWithLogitsLoss()

            history = train_model_fixed(
                model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler,
                epochs=25, patience=5, device=device, class_weights=class_weights
            )

            return history['best_val_acc']

        except Exception as e:
            print(f"Trial failed with error: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0

    # Run optimization
    print(f"\nStarting Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("\n" + "="*80)
    print("TOXIC/BOT DETECTION OPTIMIZATION RESULTS")
    print("="*80)

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")

    return {
        'study': study,
        'processed_df': processed_df,
        'tokenizer': tokenizer,
        'class_weights': class_weights
    }

# Usage examples
print("Toxic/Bot Comment Detection System Ready!")
print("\n" + "="*60)
print("USAGE EXAMPLES:")
print("="*60)
print("# Basic usage with heuristic labeling")
print("results = main_toxic_bot_detection(df, n_trials=20, labeling_method='heuristic')")
print("\n# Using unsupervised clustering for labeling")
print("results = main_toxic_bot_detection(df, n_trials=20, labeling_method='unsupervised')")
print("\n# Analyze the results")
print("processed_df = results['processed_df']")
print("toxic_comments = processed_df[processed_df['toxic_bot_label'] == 1]")
print("print('Sample toxic/bot comments:')")
print("print(toxic_comments[['comment_text', 'toxic_bot_label']].head())")
results = main_toxic_bot_detection(df, n_trials=20, labeling_method='unsupervised')
