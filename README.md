# Falcon-7B-Finetuning
---
This repository is a step-by-step approach to finetune `Falcon 7B` model on a Windows machine with a GPU.

**Project Architecture**
```
FALCON-7B-FINETUNING/
    ├── __pycache__/
    ├── data/
    │   ├── ...
    │   ├── ...
    │   ├── ...
    │   └── train-v2.0.json
    ├── env/
    ├── output_dir/
    │   └── checkpoint - X
    ├── src/
    │   ├── data/
    │   │   ├── data_preperation.py
    │   │   └── load_data.py
    │   ├── models/
    │   │   └── training.py
    │   └── utils/
    │       └── model_utils.py
    ├── trained_models/
    │   └── new_model/
    ├── .gitignore
    ├── cuda.py
    ├── question_answer.py
    ├── main.py
    ├── README.md
    └── requirements.txt
```
---
## Usage

To work with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sulaiman-shamasna/Falcon-7B-Finetuning.git
    ```
    
2. **Set up Python environment:**
    - In this project, I used **Python 3.10**.
    - Create and activate a virtual environment:
        ```bash
        python -m venv env
        ```
    - And activate it, 
      - For Windows (using Git Bash):
        ```bash
        source env/Scripts/activate
        ```
      - For Linux and macOS:
        ```bash
        source env/bin/activate
        ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    It could happen that you need to downgrade numpy version, if so, use ```NumPy version: 1.26.4```

    To ensure having CUDA installed, please run the following:
    ```
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    ```
4. **Set up CUDA/ NVIDIA Driver**
To be able to train on a GBU, first you need to have one. And to use it (on Windows), first download and install the [Nvidia Driver](https://toolbox.easeus.com/driverhandy/update-graphics-drivers-on-windows-11-nvidia.html) and the [cuDNN](https://developer.nvidia.com/cudnn-downloads?target_os=Windows&target_arch=x86_64&target_version=Agnostic&cuda_version=11).

5. **Datasets**
Download the dataset [SQuAD-explorer
](https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset) and put the json files in the directory ```./data```. In this repo, you'll find the default dir/ file(s) **```data/train-v2.0.json```**

6. **Training:**
    To run the training pipeline:
    ```bash
    python main.py
    ```
    After the training is done, you'll have your model in the directory ```/trained_models```, you can use it for the inference.

7. **Question-Answering**
    To test your trained model, make sure you spesify the right name in the **question_answer.py** file, the default is **```trained_models/new_model```**, then run:
    ```bash
    python question_answer.py
    ```
---
## Theoretical Foundations of Fine-Tuning Large Language Models
Fine-tuning large language models involves adapting a pre-trained model to a specific downstream task. The goal is to leverage the general language understanding captured during the pre-training phase and specialize the model for a particular application, such as question answering, summarization, or text classification.

### Pre-training and Fine-tuning
1. **Pre-training**: During pre-training, a large language model is trained on vast amounts of text data to learn general language representations. This phase captures syntax, semantics, and general world knowledge.

2. **Fine-tuning**: In the fine-tuning phase, the pre-trained model is further trained on a smaller, task-specific dataset. This allows the model to adapt its representations to better suit the requirements of the target task.

### Challenges in Fine-tuning Large Models
Fine-tuning large models like Falcon-7B can be computationally expensive and memory-intensive due to their sheer size. To address these challenges, researchers have developed various techniques to make fine-tuning more efficient.

## Parameter-Efficient Fine-Tuning (PEFT)
Parameter-Efficient Fine-Tuning (PEFT) techniques aim to reduce the number of trainable parameters while retaining or even improving the model's performance on downstream tasks. This is particularly important for large models, where training all parameters can be prohibitive.

## LoRA (Low-Rank Adaptation)
Low-Rank Adaptation (LoRA) is a PEFT method that modifies only a small subset of the model's parameters, specifically those in the attention mechanisms, by adding low-rank adaptations. This approach drastically reduces the number of trainable parameters and computational overhead.

### How LoRA Works
1. **Low-Rank Decomposition**: LoRA decomposes the weight matrices in the attention layers into low-rank matrices. Instead of updating the entire weight matrix, LoRA updates these smaller low-rank matrices.

2. **Efficiency**: By focusing on low-rank updates, LoRA reduces the number of trainable parameters, leading to lower memory usage and faster training times. This makes it feasible to fine-tune very large models on hardware with limited resources.

## Implementing PEFT with LoRA in Falcon-7B
The process of integrating LoRA into the fine-tuning of Falcon-7B involves several steps. Here's a more detailed look at the theoretical and practical steps:

### Step 1: Model Preparation with LoRA

**Theory**
LoRA modifies the standard Transformer architecture by introducing low-rank adaptations into the attention layers. This approach allows the model to learn task-specific information efficiently without modifying all parameters.

**Practice**
In the implementation, we use ```prepare_model_for_kbit_training``` to enable gradient checkpointing and apply LoRA configurations to the model:

``` python
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def prepare_model_for_training(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
        r=16,  # Low-rank parameter
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)
```

### Step 2: Tokenization and Dataset Preparation

**Theory**
Tokenization involves converting raw text into a format suitable for the model, typically sequences of tokens. Proper tokenization ensures that the input text is compatible with the model's vocabulary and that the sequence lengths are manageable.

**Practice**
In practice, we use the model's tokenizer to preprocess the dataset:

```python
from transformers import AutoTokenizer

def gen_prompt(text_input):
    return f"""
    <human>: {text_input["question"]}
    <assistant>: {text_input["answer"]}
    """.strip()

def gen_and_tok_prompt(text_input, tokenizer):
    full_input = gen_prompt(text_input)
    tok_full_prompt = tokenizer(full_input, padding=True, truncation=True, return_tensors="pt")
    return {
        'input_ids': tok_full_prompt['input_ids'][0],
        'attention_mask': tok_full_prompt['attention_mask'][0]
    }

def tokenize_and_check_length(batch, tokenizer):
    tokenized_batch = [gen_and_tok_prompt(x, tokenizer) for x in batch]
    input_ids_lengths = [len(x['input_ids']) for x in tokenized_batch]
    attention_mask_lengths = [len(x['attention_mask']) for x in tokenized_batch]

    # Ensure all elements have the same length
    assert all(length == input_ids_lengths[0] for length in input_ids_lengths), "Inconsistent lengths in input_ids"
    assert all(length == attention_mask_lengths[0] for length in attention_mask_lengths), "Inconsistent lengths in attention_mask"

    # Combine all input_ids and attention_mask into single tensors
    input_ids = [x['input_ids'].tolist() for x in tokenized_batch]
    attention_mask = [x['attention_mask'].tolist() for x in tokenized_batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def tokenize_dataset(data, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token
    return data.map(lambda x: tokenize_and_check_length([x], tokenizer), batched=True, remove_columns=["question", "answer"])
```

### Step 3: Fine-Tuning with CUDA
**Theory**
Fine-tuning on a GPU can significantly speed up the training process due to the parallel processing capabilities of modern GPUs. Using mixed precision training (FP16) further enhances performance by reducing memory usage and computational load.

**Practice**    
We define the training arguments and ensure that the model and data are moved to the GPU:
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=4,
    logging_steps=25,
    output_dir="output_dir",  # Location to store checkpoints
    save_strategy='epoch',
    optim="paged_adamw_8bit",
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
)

model = model.to("cuda" if torch.cuda.is_available() else "cpu")

from training import fine_tune_model

fine_tune_model(model, tokenizer, data, training_args)
```

### Step 4: Saving and Using the Trained Model
**Theory**
Saving the model allows you to reuse the fine-tuned weights for inference or further training without having to repeat the entire fine-tuning process. This is crucial for practical deployment and iterative development.

**Practice**
After training, save the model:
```python
from model_utils import save_model

save_model(model, 'trained_models/NAME-YOUR-MODEL')
```
---