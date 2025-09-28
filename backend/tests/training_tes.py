# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import pickle
from typing import Dict, List, Tuple, Any
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Data Preprocessing
class BloomDataPreprocessor:
    """
    Preprocessor untuk membersihkan dan memproses data teks soal ujian
    """
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text: str) -> str:
        """
        Membersihkan teks dari noise dan normalisasi
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, email, and special patterns
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional, bisa dipertahankan jika relevan)
        # text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenisasi dan lemmatization dengan filtering stopwords
        
        Args:
            text: Cleaned text input
            
        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
                
        return processed_tokens
    
    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess seluruh dataset
        
        Args:
            df: DataFrame dengan kolom 'Questions' dan 'Category'
            
        Returns:
            Preprocessed DataFrame
        """
        print("🔄 Preprocessing dataset...")
        
        # Copy dataframe
        processed_df = df.copy()
        
        # Clean text
        processed_df['cleaned_question'] = processed_df['Questions'].apply(self.clean_text)
        
        # Tokenize and create processed text
        processed_df['tokens'] = processed_df['cleaned_question'].apply(self.tokenize_and_lemmatize)
        processed_df['processed_text'] = processed_df['tokens'].apply(lambda x: ' '.join(x))
        
        # Encode labels
        processed_df['encoded_label'] = self.label_encoder.fit_transform(processed_df['Category'])
        
        # Remove empty processed texts
        processed_df = processed_df[processed_df['processed_text'].str.len() > 0].reset_index(drop=True)
        
        print(f"✅ Preprocessing completed. {len(processed_df)} samples ready.")
        print(f"📊 Label distribution:\n{processed_df['Category'].value_counts()}")
        
        return processed_df

# Keyword Extraction using TF-IDF
class KeywordExtractor:
    """
    Ekstraksi keyword menggunakan TF-IDF dan Mutual Information filtering
    """
    
    def __init__(self, max_features: int = 5000, top_n_keywords: int = 50):
        self.max_features = max_features
        self.top_n_keywords = top_n_keywords
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # unigram dan bigram
            min_df=2,  # minimal muncul di 2 dokumen
            max_df=0.95  # maksimal di 95% dokumen
        )
        self.selected_features = None
        self.feature_names = None
        
    def extract_keywords(self, texts: List[str], labels: np.ndarray) -> np.ndarray:
        """
        Ekstraksi keyword dengan TF-IDF dan filtering dengan Mutual Information
        
        Args:
            texts: List of processed text
            labels: Encoded labels array
            
        Returns:
            Keyword feature matrix
        """
        print("🔍 Extracting keywords with TF-IDF...")
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        print(f"📈 Initial TF-IDF features: {tfidf_matrix.shape[1]}")
        
        # Apply Mutual Information for feature selection
        print("🧠 Applying Mutual Information for keyword filtering...")
        mi_scores = mutual_info_classif(tfidf_matrix, labels, random_state=42)
        
        # Select top keywords based on MI scores
        top_indices = np.argsort(mi_scores)[-self.top_n_keywords:]
        self.selected_features = top_indices
        
        # Create filtered keyword matrix
        keyword_matrix = tfidf_matrix[:, top_indices].toarray()
        
        selected_keywords = [self.feature_names[i] for i in top_indices]
        print(f"✅ Selected {len(selected_keywords)} high-MI keywords")
        print(f"🏷️ Top 10 keywords: {selected_keywords[-10:]}")
        
        return keyword_matrix
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform teks baru menggunakan keyword extractor yang sudah di-fit
        """
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        return tfidf_matrix[:, self.selected_features].toarray()

# Dataset Class for PyTorch
class BloomDataset(Dataset):
    """
    PyTorch Dataset untuk data Bloom Taxonomy
    """
    
    def __init__(self, questions: List[str], keyword_features: np.ndarray, 
                 labels: np.ndarray, tokenizer, max_length: int = 256):
        self.questions = questions
        self.keyword_features = torch.FloatTensor(keyword_features)
        self.labels = torch.LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = str(self.questions[idx])
        
        # Tokenize dengan RoBERTa tokenizer
        encoding = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'keyword_features': self.keyword_features[idx],
            'labels': self.labels[idx]
        }

# Bloom Classifier Model
class BloomClassifier(nn.Module):
    """
    Model gabungan LSTM + CNN untuk semantic embedding + Keyword features
    """
    
    def __init__(self, roberta_model_name: str = 'roberta-base', 
                 keyword_dim: int = 50, num_classes: int = 5,
                 lstm_hidden_dim: int = 128, cnn_filters: int = 100,
                 dropout_rate: float = 0.3):
        super(BloomClassifier, self).__init__()
        
        # RoBERTa untuk semantic embedding
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size  # 768 untuk roberta-base
        
        # Freeze beberapa layer RoBERTa untuk efisiensi (optional)
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        
        # LSTM untuk sequential processing
        self.lstm = nn.LSTM(
            input_size=self.roberta_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # CNN untuk local pattern detection
        self.conv1d = nn.Conv1d(
            in_channels=self.roberta_dim,
            out_channels=cnn_filters,
            kernel_size=3,
            padding=1
        )
        
        # Pooling layers
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature fusion dimensions
        lstm_output_dim = lstm_hidden_dim * 2  # bidirectional
        cnn_output_dim = cnn_filters * 2  # max + avg pooling
        combined_semantic_dim = lstm_output_dim + cnn_output_dim
        
        # MLP untuk klasifikasi final
        self.classifier = nn.Sequential(
            nn.Linear(combined_semantic_dim + keyword_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),  
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, input_ids, attention_mask, keyword_features):
        # RoBERTa semantic embedding
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = roberta_output.last_hidden_state  # [batch, seq_len, hidden_dim]
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        # Ambil hidden state terakhir dari kedua arah
        lstm_features = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, lstm_hidden_dim*2]
        
        # CNN processing
        # Transpose untuk conv1d: [batch, hidden_dim, seq_len]
        cnn_input = sequence_output.transpose(1, 2)
        cnn_output = torch.relu(self.conv1d(cnn_input))  # [batch, filters, seq_len]
        
        # Global pooling
        cnn_max = self.global_max_pool(cnn_output).squeeze(-1)  # [batch, filters]
        cnn_avg = self.global_avg_pool(cnn_output).squeeze(-1)  # [batch, filters]
        cnn_features = torch.cat([cnn_max, cnn_avg], dim=1)  # [batch, filters*2]
        
        # Combine semantic features
        semantic_features = torch.cat([lstm_features, cnn_features], dim=1)
        semantic_features = self.dropout(semantic_features)
        
        # Concatenate dengan keyword features
        combined_features = torch.cat([semantic_features, keyword_features], dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        
        return logits

# Trainer Class
class BloomTrainer:
    """
    Training pipeline untuk Bloom Classifier
    """
    
    def __init__(self, model, device, learning_rate: float = 2e-5):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=3, factor=0.5, verbose=True
        )
        
    def train_epoch(self, dataloader):
        """Training untuk satu epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            keyword_features = batch['keyword_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, keyword_features)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        """Evaluasi model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                keyword_features = batch['keyword_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, keyword_features)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels

# Optuna Hyperparameter Optimization
class OptunaObjective:
    def __init__(self, train_data, val_data, test_data, num_classes, device, keyword_dim, tokenizer, max_length):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_classes = num_classes
        self.device = device
        self.keyword_dim = keyword_dim
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, trial: optuna.Trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        num_epochs = trial.suggest_int('num_epochs', 5, 20)
        lstm_hidden_dim = trial.suggest_categorical('lstm_hidden_dim', [64, 128, 256])
        cnn_filters = trial.suggest_categorical('cnn_filters', [50, 100, 150])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)

        # Prepare datasets and dataloaders
        train_dataset = BloomDataset(
            self.train_data['questions'],
            self.train_data['keyword_features'],
            self.train_data['labels'],
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = BloomDataset(
            self.val_data['questions'],
            self.val_data['keyword_features'],
            self.val_data['labels'],
            self.tokenizer,
            self.max_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize the model with the suggested hyperparameters
        model = BloomClassifier(
            roberta_model_name='roberta-base',
            keyword_dim=self.keyword_dim,
            num_classes=self.num_classes,
            lstm_hidden_dim=lstm_hidden_dim,
            cnn_filters=cnn_filters,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        trainer = BloomTrainer(model, self.device, learning_rate)
        
        best_val_accuracy = 0
        for epoch in range(num_epochs):
            train_loss, train_acc = trainer.train_epoch(train_loader)
            val_loss, val_acc, val_predictions, val_labels = trainer.evaluate(val_loader)

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc

            trial.report(val_acc, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_accuracy

# Run Optuna optimization
def run_optuna_optimization(train_data, val_data, test_data, num_classes, device, keyword_dim, tokenizer, max_length, n_trials=50):
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    objective = OptunaObjective(
        train_data, val_data, test_data,
        num_classes, device, keyword_dim,
        tokenizer, max_length
    )
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (accuracy): {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study.best_params, study

def main():
    print("🚀 Starting Bloom Taxonomy Classification Pipeline with Optuna")
    
    CONFIG = {
        'test_size': 0.2,
        'val_size': 0.125,
        'max_length': 256,
        'max_features': 5000,
        'top_n_keywords': 50,
        'random_seed': 42,
        'n_trials': 50
    }
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    # Load and preprocess data
    df = pd.read_csv('/kaggle/input/blooms-taxonomy-dataset/blooms_taxonomy_dataset.csv')
    preprocessor = BloomDataPreprocessor()
    processed_df = preprocessor.preprocess_dataset(df)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        processed_df[['Questions', 'processed_text']],
        processed_df[['Category', 'encoded_label']],
        test_size=CONFIG['test_size'],
        stratify=processed_df['encoded_label'],
        random_state=CONFIG['random_seed']
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=CONFIG['val_size'],
        stratify=y_temp['encoded_label'],
        random_state=CONFIG['random_seed']
    )
    
    # Extract keywords
    keyword_extractor = KeywordExtractor(
        max_features=CONFIG['max_features'],
        top_n_keywords=CONFIG['top_n_keywords']
    )
    train_keyword_features = keyword_extractor.extract_keywords(
        X_train['processed_text'].tolist(),
        y_train['encoded_label'].values
    )
    val_keyword_features = keyword_extractor.transform(X_val['processed_text'].tolist())
    test_keyword_features = keyword_extractor.transform(X_test['processed_text'].tolist())
    
    # Initialize tokenizer and set up the number of classes
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    num_classes = len(processed_df['Category'].unique())
    
    # Run Optuna optimization
    best_params, study = run_optuna_optimization(
        {'questions': X_train['Questions'].tolist(), 'keyword_features': train_keyword_features, 'labels': y_train['encoded_label'].values},
        {'questions': X_val['Questions'].tolist(), 'keyword_features': val_keyword_features, 'labels': y_val['encoded_label'].values},
        {'questions': X_test['Questions'].tolist(), 'keyword_features': test_keyword_features, 'labels': y_test['encoded_label'].values},
        num_classes, device, CONFIG['top_n_keywords'], tokenizer, CONFIG['max_length'], n_trials=CONFIG['n_trials']
    )

    # Now use the best hyperparameters found by Optuna for the final model
    print("🎯 Finalizing training with the best hyperparameters...")
    # Additional steps to train the final model using the best hyperparameters...

if __name__ == "__main__":
    main()