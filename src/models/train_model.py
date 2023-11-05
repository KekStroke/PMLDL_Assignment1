import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
from datasets import load_metric
from datasets import Dataset, DatasetDict
import torch
import os
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

MAX_LENGTH = 85
BATCH_SIZE = 64
checkpoint_name = "test"
train_ratio = 0.8
val_test_ratio = 0.5

# Set model name
model_name = 'SkolkovoInstitute/bart-base-detox'

# Read preprocessed dataset
df = pd.read_csv('data/internal/preprocessed.csv', index_col=0)

# Get tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=MAX_LENGTH, cache_dir=f".cache/tokenizers/{model_name}")

VOCAB_SIZE = len(tokenizer)

# Split dataset into train, validation and test datasets
ref_train, ref_val, trn_train, trn_val = train_test_split(df['reference'].tolist(), df['translation'].tolist(), test_size=(1-train_ratio), random_state=42)
ref_val, ref_test, trn_val, trn_test = train_test_split(ref_val, trn_val, test_size=val_test_ratio, random_state=42)

# Create custom HuggingFace dataset
train_data = pd.DataFrame({'input_text': ref_train, 'target_text': trn_train})
val_data = pd.DataFrame({'input_text': ref_val, 'target_text': trn_val})
test_data = pd.DataFrame({'input_text': ref_test, 'target_text': trn_test})
raw_datasets = DatasetDict({
    'train': Dataset.from_pandas(train_data),
    'validation': Dataset.from_pandas(val_data),
    'test': Dataset.from_pandas(test_data)
})

prefix = ""
source_lang = "input_text"
target_lang = "target_text"

# Function to preprocess reference and target text
def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples['input_text']]
    targets = [ex for ex in examples['target_text']]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess datasets
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# Get model checkpoints
load_ckpt_path = f'models/{model_name}'
model_ckpt_path = load_ckpt_path+'/' + checkpoint_name
ckpt_path = model_ckpt_path+'_checkpoint.pt'

isCkptExists = os.path.isdir(model_ckpt_path)

if not isCkptExists:
    print('Checkpoint file does not exist. Training model from scratch!')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=f".cache/models/{model_name}")
    val_scores = []
    best = 0

    torch.save({
      'val_scores': val_scores,
      }, ckpt_path)
elif isCkptExists:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt_path)
    
# Set training arguments
args = Seq2SeqTrainingArguments(
    f"models/{model_name}",
    evaluation_strategy = "steps",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
)

# Implement the batch creation for training
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

blue_metric = load_metric("sacrebleu")

# simple postprocessing for text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# compute metrics function to pass to trainer
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = blue_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    checkpoint = torch.load(ckpt_path)
    val_scores = checkpoint['val_scores']

    val_scores.append(result["bleu"])

    torch.save({
      'val_scores': val_scores,
      }, ckpt_path)

    result = {k: round(v, 4) for k, v in result.items()}
    return result

# instead of writing train loop we will use Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

# Save model
trainer.save_model(f'models/{model_name}/{checkpoint_name}_new')