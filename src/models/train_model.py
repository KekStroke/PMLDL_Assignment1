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


def parse_arguments():
    """
    Parse command-line arguments and set default values.
    Args:
        None
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # Default values
    max_length = 85
    batch_size = 64
    checkpoint_name = "test"
    model_name = 'SkolkovoInstitute/bart-base-detox'
    train_ratio = 0.8
    val_test_ratio = 0.5
    learning_rate=2e-5
    weight_decay=0.01
    save_total_limit=1
    num_train_epochs=1
    save_steps=500
    eval_steps=500
    logging_steps=100
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", nargs='?', const=max_length, type=int, default=max_length)
    parser.add_argument("--batch_size", nargs='?', const=batch_size, type=int, default=batch_size)
    parser.add_argument("--checkpoint_name", nargs='?', const=checkpoint_name, type=str, default=checkpoint_name)
    parser.add_argument("--train_ratio", nargs='?', const=train_ratio, type=float, default=train_ratio)
    parser.add_argument("--val_test_ratio", nargs='?', const=val_test_ratio, type=float, default=val_test_ratio)
    parser.add_argument("--model_name", nargs='?', const=model_name, type=str, default=model_name)
    parser.add_argument("--learning_rate", nargs='?', const=learning_rate, type=float, default=learning_rate)
    parser.add_argument("--weight_decay", nargs='?', const=weight_decay, type=float, default=weight_decay)
    parser.add_argument("--save_total_limit", nargs='?', const=save_total_limit, type=int, default=save_total_limit)
    parser.add_argument("--num_train_epochs", nargs='?', const=num_train_epochs, type=int, default=num_train_epochs)
    parser.add_argument("--save_steps", nargs='?', const=save_steps, type=int, default=save_steps)
    parser.add_argument("--eval_steps", nargs='?', const=eval_steps, type=int, default=eval_steps)
    parser.add_argument("--logging_steps", nargs='?', const=logging_steps, type=int, default=logging_steps)
    
    args = parser.parse_args()
    return args

def set_seed():
    """
    Set random seeds for reproducibility.
    """
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

def read_dataset():
    """
    Read the preprocessed dataset from a CSV file.
    Args:
        None
    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    return pd.read_csv('data/internal/preprocessed.csv', index_col=0)

def create_tokenizer(model_name, max_length):
    """
    Create a tokenizer from the Hugging Face model and set model parameters.
    Args:
        model_name (str): The Hugging Face model name.
        max_length (int): The maximum amount of tokens.
    Returns:
        transformers.AutoTokenizer: The created tokenizer.
    """
    return AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        cache_dir=f".cache/tokenizers/{model_name}"
    )

def split_dataset(df, train_ratio, val_test_ratio):
    """
    Split the dataset into train, validation, and test datasets.
    Args:
        df (pd.DataFrame): The preprocessed dataset.
        train_ratio (float): Ratio of data used for training.
        val_test_ratio (float): Ratio of validation data within the remaining data.
    Returns:
        pd.Series: Splits of reference and translation for training, validation, and testing.
    """
    ref_train, ref_val, trn_train, trn_val = train_test_split(df['reference'].tolist(), df['translation'].tolist(), test_size=(1 - train_ratio), random_state=42)
    ref_val, ref_test, trn_val, trn_test = train_test_split(ref_val, trn_val, test_size=val_test_ratio, random_state=42)
    return ref_train, ref_val, trn_train, trn_val, ref_test, trn_test

def create_datasets(ref_train, ref_val, trn_train, trn_val, ref_test, trn_test):
    """
    Create custom HuggingFace datasets for training, validation, and testing.
    Args:
        ref_train (list): Reference data for training.
        ref_val (list): Reference data for validation.
        trn_train (list): Translation data for training.
        trn_val (list): Translation data for validation.
        ref_test (list): Reference data for testing.
        trn_test (list): Translation data for testing.
    Returns:
        datasets.DatasetDict: Custom datasets for training, validation, and testing.
    """
    train_data = pd.DataFrame({'input_text': ref_train, 'target_text': trn_train})
    val_data = pd.DataFrame({'input_text': ref_val, 'target_text': trn_val})
    test_data = pd.DataFrame({'input_text': ref_test, 'target_text': trn_test})
    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train_data),
        'validation': Dataset.from_pandas(val_data),
        'test': Dataset.from_pandas(test_data)
    })
    return raw_datasets


# Function to preprocess reference and target text
def preprocess_function(examples, max_length, model_name):
    """
    Preprocesses the input and target text examples.

    Args:
        examples (dict): A dictionary containing 'input_text' and 'target_text' fields.
        max_length (int): The maximum amount of tokens.

    Returns:
        dict: Model inputs with tokenized and processed data.
    """
    inputs = [ex for ex in examples['input_text']]
    targets = [ex for ex in examples['target_text']]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Function to load or create the model
def load_or_create_model(model_name, checkpoint_name):
    """
    Loads an existing model checkpoint or creates a new model if the checkpoint doesn't exist.

    Args:
        model_name (str): The model name for loading or initializing.
        checkpoint_name (str): The checkpoint name.

    Returns:
        model: The loaded or initialized model.
    """
    global load_ckpt_path, model_ckpt_path, ckpt_path
    
    load_ckpt_path = f'models/{model_name}'
    model_ckpt_path = load_ckpt_path + '/' + checkpoint_name
    ckpt_path = model_ckpt_path + '_checkpoint.pt'

    isCkptExists = os.path.isdir(model_ckpt_path)

    if not isCkptExists:
        print('Checkpoint file does not exist. Training model from scratch!')
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=f".cache/models/{model_name}")
        val_scores = []
        
        torch.save({'val_scores': val_scores}, ckpt_path)
    elif isCkptExists:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt_path)

    return model

def set_training_args(model_name, learning_rate, batch_size, weight_decay, save_total_limit, num_train_epochs, save_steps, eval_steps, logging_steps):
    """
    Set training arguments for Seq2Seq model.

    Args:
        model_name (str): Name of the model.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size for training and evaluation.
        weight_decay (float): Weight decay.
        save_total_limit (int): Total number of checkpoints to save.
        num_train_epochs (int): Number of training epochs.
        save_steps (int): Number of steps between model checkpoints.
        eval_steps (int): Number of steps between evaluations.
        logging_steps (int): Number of steps between logging.

    Returns:
        Seq2SeqTrainingArguments: Training arguments.
    """
    args = Seq2SeqTrainingArguments(
        f"models/{model_name}",
        evaluation_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        fp16=True,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
    )
    return args

def postprocess_text(preds, labels):
    """
    Postprocess predicted and reference texts.

    Args:
        preds (list): Predicted texts.
        labels (list): Reference texts.

    Returns:
        tuple: Processed predicted and reference texts.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds, tokenizer, ckpt_path):
    """
    Compute evaluation metrics for Seq2Seq model.

    Args:
        eval_preds (tuple): Tuple containing predicted and reference texts.
        tokenizer: Tokenizer for decoding.
        ckpt_path (str): Checkpoint path for model.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    blue_metric = load_metric("sacrebleu")
    result = blue_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)

    checkpoint = torch.load(ckpt_path)
    val_scores = checkpoint['val_scores']
    val_scores.append(result["bleu"])

    torch.save({'val_scores': val_scores}, ckpt_path)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

def train_seq2seq_model(model, args, train_dataset, eval_dataset, data_collator, tokenizer, compute_metrics):
    """
    Train a Seq2Seq model.

    Args:
        model: Seq2Seq model to train.
        args: Training arguments.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        data_collator: Data collator.
        tokenizer: Tokenizer for decoding.
        compute_metrics: Function to compute evaluation metrics.

    Returns:
        Seq2SeqTrainer: Trained Seq2Seq model.
    """
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer

def save_model(trainer, model_name, checkpoint_name):
    """
    Save a model using a trainer.

    Args:
        trainer (Trainer): The trainer object responsible for training the model.
        model_name (str): The name of the model.
        checkpoint_name (str): The name of the checkpoint.

    Returns:
        None
    """
    model_path = f'models/{model_name}/{checkpoint_name}_new'
    trainer.save_model(model_path)

if __name__ == "__main__":
    
    args = parse_arguments()
    
    set_seed()
    
    df = read_dataset()
    tokenizer = create_tokenizer(args.model_name, args.max_length)
    ref_train, ref_val, trn_train, trn_val, ref_test, trn_test = split_dataset(df, args.train_ratio, args.val_test_ratio)
    raw_datasets = create_datasets(ref_train, ref_val, trn_train, trn_val, ref_test, trn_test)
    
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, fn_kwargs={'max_length':args.max_length, 'model_name':args.model_name})
    
    model = load_or_create_model(args.model_name, args.checkpoint_name)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    trainargs = set_training_args(args.model_name, args.learning_rate, args.batch_size, args.weight_decay, args.save_total_limit, args.num_train_epochs, args.save_steps, args.eval_steps, args.logging_steps)
    trainer = train_seq2seq_model(model, trainargs, tokenized_datasets["train"], tokenized_datasets["validation"], data_collator, tokenizer, compute_metrics)
    
    save_model(trainer, args.model_name, args.checkpoint_name)