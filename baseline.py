import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from evaluate import load
import numpy as np
from tqdm import tqdm
from sari import compute_sari_components

# Load model and tokenizer
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load bertscore
bert_score = load("bertscore")

"""
Simplify input text using the model and return the simplified text.

Args:
    text (str): The input text to simplify.
    max_length (int): The maximum length of the output text.
"""
def simplify_text(text, max_length=128):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

"""
Evaluate the simplification model on the given DataFrame and return the evaluation results.

Args:
    df (pd.DataFrame): The DataFrame containing the 'complex' and 'simple' columns.
"""
def evaluate_simplification(df):
    results = {
        'bert_scores': [],
        'sari_scores': [],
        'add_scores': [],
        'keep_scores': [],
        'del_scores': []
    }
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        complex_sent = row['complex']
        simple_sent = row['simple']
        
        predicted_simple = simplify_text(complex_sent)
        
        bert_output = bert_score.compute(
            predictions=[predicted_simple],
            references=[simple_sent],
            lang="en"
        )
        results['bert_scores'].append(np.mean(bert_output['f1']))
        
        sari_components = compute_sari_components(complex_sent, predicted_simple, simple_sent)
        results['sari_scores'].append(sari_components['sari'])
        results['add_scores'].append(sari_components['add_score'])
        results['keep_scores'].append(sari_components['keep_score'])
        results['del_scores'].append(sari_components['deletion_score'])
    
    return results


df_validation = pd.read_csv('data/medical/validation.csv')
results = evaluate_simplification(df_validation)

print(f"BERTScore: {results['bert_scores']:.4f}")
print(f"SARI Score: {results['sari_scores']:.4f}")
print(f"Add Score: {results['add_scores']:.4f}")
print(f"Keep Score: {results['keep_scores']:.4f}")
print(f"Delete Score: {results['del_scores']:.4f}")