from datasets import load_dataset
import pandas as pd
import os

# Load dataset (you can use 10% for faster testing)
ds = load_dataset("civil_comments", split="train[:10%]")
df = ds.to_pandas()

# Categorization function
def categorize_comment(row):
    text = str(row['text']).lower() if pd.notnull(row['text']) else ""
    
    if row.get('toxicity', 0) > 0.8 or "trash" in text:
        return "Hate/Abuse"
    elif row.get('severe_toxicity', 0) > 0.5:
        return "Threat"
    elif "follow me" in text or "subscribe" in text:
        return "Spam"
    elif "can you" in text or "could you" in text or "would you" in text:
        return "Question/Suggestion"
    elif "great" in text or "amazing" in text:
        return "Praise"
    elif "keep going" in text or "you’re doing great" in text:
        return "Support"
    elif "but" in text and ("okay" in text or "not bad" in text):
        return "Constructive Criticism"
    elif "reminded me" in text or "crying" in text:
        return "Emotional"
    else:
        return "Other"

# Apply categorization
df['label'] = df.apply(categorize_comment, axis=1)

# Keep only the necessary columns
df = df[['text', 'label']]

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save to CSV
df.to_csv("data/processed_comments.csv", index=False)
print("✅ Processed comments saved to data/processed_comments.csv")
