import wandb
import torch
import numpy as np
import pandas as pd
from evaluate import load
from sari import compute_sari_components
from transformers import Trainer, TrainingArguments
from classes.simplification_dataset import SimplificationDataset
from transformers import BartTokenizer, BartForConditionalGeneration

"""
Calculate the BERTScore and SARI scores for the given texts and references.

Args:
    model (BartForConditionalGeneration): The model to use for generating the predictions.
    tokenizer (BartTokenizer): The tokenizer to use for tokenizing the inputs.
    texts (List[str]): The input texts to generate predictions for.
    references (List[str]): The reference texts to compare the predictions against.
    device (torch.device): The device to run the model on.
"""
def compute_metrics(model, tokenizer, texts, references, device):
    predictions = []
    bert_score = load("bertscore")
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4)
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    bert_results = bert_score.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )
    
    sari_components = [
        compute_sari_components(texts[i], predictions[i], references[i])
        for i in range(len(texts))
    ]
    
    return {
        'bert_score': np.mean(bert_results['f1']),
        'sari': np.mean([s['sari'] for s in sari_components]),
        'add_score': np.mean([s['add_score'] for s in sari_components]),
        'keep_score': np.mean([s['keep_score'] for s in sari_components]),
        'del_score': np.mean([s['deletion_score'] for s in sari_components])
    }

"""
Train and evaluate the model using the given Weights & Biases configuration.

Args:
    config (Dict): The configuration for the model training.
"""
def train_and_evaluate(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        train_df = pd.read_csv('data/medical/train.csv')
        val_df = pd.read_csv('data/medical/validation.csv')
        
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Log initial metrics
        initial_metrics = compute_metrics(
            model, tokenizer,
            val_df['complex'].tolist(),
            val_df['simple'].tolist(),
            device
        )
        wandb.log({f"initial_{k}": v for k, v in initial_metrics.items()})
        
        train_dataset = SimplificationDataset(train_df, tokenizer)
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
        # Log final metrics
        final_metrics = compute_metrics(
            model, tokenizer,
            val_df['complex'].tolist(),
            val_df['simple'].tolist(),
            device
        )
        wandb.log({f"final_{k}": v for k, v in final_metrics.items()})
        
        return final_metrics['sari']
    

sweep_config = {
    'method': 'random',
    'metric': {'name': 'final_sari', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [1e-5, 2e-5, 5e-5]},
        'batch_size': {'values': [4, 8]},
        'num_epochs': {'values': [2, 3]},
        'warmup_steps': {'values': [100, 250, 500]},
        'gradient_accumulation_steps': {'values': [1, 2]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="text-simplification")
wandb.agent(sweep_id, train_and_evaluate)