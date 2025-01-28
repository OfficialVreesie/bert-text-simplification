from torch.utils.data import Dataset

class SimplificationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.complex_texts = df['complex'].tolist()
        self.simple_texts = df['simple'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.complex_texts)

    def __getitem__(self, idx):
        complex_text = self.complex_texts[idx]
        simple_text = self.simple_texts[idx]

        inputs = self.tokenizer(
            complex_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            simple_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }