from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

def fine_tune_model(model, tokenizer, data, training_args):
    training_args = TrainingArguments(**training_args)
    
    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False

    if torch.cuda.is_available():
        model.cuda()
    
    trainer.train()