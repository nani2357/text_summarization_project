import os
from pathlib import Path
from transformers import AutoTokenizer
from textSummarization.logging import logger
from textSummarization.entity import DataTransformationConfig
from datasets import load_dataset, load_from_disk



class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
        
    def convert_example_to_features(self, df):
        input_encodings = self.tokenizer(df['dialogue'], max_length=1024, truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(df['summary'], max_length=128, truncation=True)
            
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }        
        
        
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_example_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))