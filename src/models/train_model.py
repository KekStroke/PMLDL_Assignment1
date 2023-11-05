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
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length")
    parser.add_argument("--batch_size")
    parser.add_argument("--checkpoint_name")
    parser.add_argument("--train_ratio")
    parser.add_argument("--val_test_ratio")
    parser.add_argument("--model_name")
    parser.add_argument("--learning_rate")
    parser.add_argument("--weight_decay")
    parser.add_argument("--save_total_limit")
    parser.add_argument("--num_train_epochs")
    parser.add_argument("--save_steps")
    parser.add_argument("--eval_steps")
    parser.add_argument("--logging_steps")
    
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    MAX_LENGTH = 85
    BATCH_SIZE = 64
    checkpoint_name = "test"
    train_ratio = 0.8
    val_test_ratio = 0.5
    
    learning_rate=2e-5
    weight_decay=0.01
    save_total_limit=1
    num_train_epochs=10
    save_steps=500
    eval_steps=500
    logging_steps=100
    
    if (args.max_length is not None): 
        MAX_LENGTH = args.max_length
    if (args.batch_size is not None): 
        BATCH_SIZE = args.batch_size
    if (args.checkpoint_name is not None): 
        checkpoint_name = args.checkpoint_name
    if (args.train_ratio is not None): 
        train_ratio = args.train_ratio
    if (args.val_test_ratio is not None): 
        val_test_ratio = args.val_test_ratio
    if (args.learning_rate is not None): 
        learning_rate = args.learning_rate
    if (args.weight_decay is not None): 
        weight_decay = args.weight_decay
    if (args.save_total_limit is not None): 
        save_total_limit = args.save_total_limit
    if (args.num_train_epochs is not None): 
        num_train_epochs = args.num_train_epochs
    if (args.save_steps is not None): 
        save_steps = args.save_steps
    if (args.eval_steps is not None): 
        eval_steps = args.eval_steps
    if (args.logging_steps is not None): 
        logging_steps = args.logging_steps

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
        learning_rate=learning_rate,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=True,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
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