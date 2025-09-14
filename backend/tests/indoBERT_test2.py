import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TaxonomyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model():
    # Load dataset
    df = pd.read_csv('dataset (1).csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Taksonomi'].value_counts()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['Taksonomi'])
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['pertanyaan'], df['label_encoded'], 
        test_size=0.2, random_state=42, stratify=df['label_encoded']
    )
    
    # Load IndoBERT tokenizer and model
    model_name = 'indobenchmark/indobert-base-p1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    num_labels = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.to(device)
    
    # Create datasets
    train_dataset = TaxonomyDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
    val_dataset = TaxonomyDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training parameters
    epochs = 5
    learning_rate = 2e-5
    num_training_steps = len(train_loader) * epochs
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_accuracy = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Training
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_true_labels, val_predictions)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'label_encoder': label_encoder,
            }, 'best_model.pth')
            print("Saved best model!")
    
    # Load best model for final evaluation
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    final_accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"\nFinal Validation Accuracy: {final_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=label_encoder.classes_
    ))
    
    # Save the complete model and tokenizer
    model.save_pretrained('./taxonomy_classifier')
    tokenizer.save_pretrained('./taxonomy_classifier')
    
    # Save label encoder
    import joblib
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    print("\nModel and tokenizer saved successfully!")
    
    return model, tokenizer, label_encoder

def predict_taxonomy(text, model, tokenizer, label_encoder):
    """Function to predict taxonomy for new text"""
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    
    return label_encoder.inverse_transform([prediction])[0]

if __name__ == "__main__":
    # Train the model
    print("Starting model training...")
    model, tokenizer, label_encoder = train_model()
    
    # Example prediction
    test_text = "Penulisan ejaan kata yang bercetak miring pada paragraf tersebut adalah"
    prediction = predict_taxonomy(test_text, model, tokenizer, label_encoder)
    print(f"\nExample prediction for: '{test_text}'")
    print(f"Predicted taxonomy: {prediction}")