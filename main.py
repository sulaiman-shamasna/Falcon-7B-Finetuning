import sys
import pandas as pd
from datasets import Dataset
from src.data.data_preparation import tokenize_dataset

from src.utils.model_utils import (load_model_and_tokenizer,
                                    prepare_model_for_training,
                                    save_model,
                                    print_trainable_parameters)

from src.models.training import fine_tune_model
from src.data.load_data import df_squad

import numpy as np
import torch
from datetime import datetime

print(f"NumPy version: {np.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")

df_faq = df_squad

# df_faq = pd.DataFrame({
#     'question': ['What is AI?', 'How to train a model?'],
#     'answer': ['AI is the simulation of human intelligence in machines.', 'Training a model involves feeding it data and adjusting its parameters.']
# })

data = Dataset.from_pandas(df_faq[['question', 'answer']])

model_name = "tiiuae/falcon-7b-instruct"
model, tokenizer = load_model_and_tokenizer(model_name)

if torch.cuda.is_available():
    model.cuda()

data = tokenize_dataset(data, tokenizer)

model = prepare_model_for_training(model)
print_trainable_parameters(model)

training_args = {
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "save_total_limit": 4,
    "logging_steps": 25,
    "output_dir": "output_dir",  # give the location where you want to store checkpoints
    "save_strategy": 'epoch',
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": 'cosine',
    "warmup_ratio": 0.05,
    "per_device_train_batch_size": 1,  # Adjust based on your GPU memory
    # "device": "cuda" if torch.cuda.is_available() else "cpu"
}

fine_tune_model(model, tokenizer, data, training_args)

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

save_model(model, f'trained_models/MODEL_{current_time}')
