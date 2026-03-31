import pandas as pd
import json

import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the JSON file relative to the script location
json_path = os.path.join(script_dir, "tatqa_dataset_dev.json")

# Use json_path when opening the file
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

chunks = []
for entry in data:
    # Paragraphs
    for para in entry.get('paragraphs', []):
        chunks.append({'type': 'paragraph', 'text': para['text']})
    # Questions (and answers if available)
    for idx, question in enumerate(entry.get('questions', [])):
        # You may need to adjust this if answers are stored elsewhere
        chunks.append({
            'type': 'question',
            'question': question.get('question', ''),  # adjust key as needed
            'answer': question.get('answer', ''),      # adjust key as needed
            'scale': question.get('scale', ''),
            'context': entry.get('paragraphs', [{}])[0].get('text', '')  # or better context mapping
        })
    pass

pd.DataFrame(chunks).to_csv(os.path.join(script_dir, 'tatqa_chunks.csv'), index=False)