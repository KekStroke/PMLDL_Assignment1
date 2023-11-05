import pandas as pd
from transformers import RobertaTokenizer
import argparse

def read_and_preprocess_data(input_file):
    """
    Read a dataset, preprocess it, and return the preprocessed DataFrame.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Read dataset and remove unnecessary column
    df = pd.read_csv(input_file, sep='\t', index_col=0)
    df = df.drop('lenght_diff', axis=1)

    # Swap columns with ref_tox and trn_tox mixed up
    df_mixed_up = df[(df['ref_tox'] < df['trn_tox'])]
    df_mixed_up.columns = ['translation', 'reference', 'similarity', 'trn_tox', 'ref_tox']
    df_proper = df.copy()
    df_proper.loc[df_mixed_up.index] = df_mixed_up.loc[df_mixed_up.index]

    # Create ref len and trn len columns
    df_proper['reference_length'] = df_proper['reference'].apply(lambda x: len(x.split()))
    df_proper['translation_length'] = df_proper['translation'].apply(lambda x: len(x.split()))

    return df_proper

def filter_data(df):
    """
    Filter the DataFrame to select specific rows based on conditions.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Get only rows with very toxic reference and with very low toxicity translation
    df_similar = df[(df['ref_tox'] > 0.99) & (df['trn_tox'] < 0.01) & (df['similarity'] > 0.7) & (df['reference_length'] <= 60) & (df['translation_length'] <= 60)]
    return df_similar

def tokenize_data(df, tokenizer_model):
    """
    Tokenize the reference and translation columns of the DataFrame using a specified tokenizer.

    Args:
        df (pd.DataFrame): Filtered DataFrame.
        tokenizer_model (str): Name of the tokenizer model to use.

    Returns:
        pd.DataFrame: DataFrame with tokenized columns.
    """
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_model, cache_dir=".cache/tokenizers/roberta_toxicity_classifier")
    
    df.loc[:, 'tokenized_reference'] = df.loc[:, 'reference'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    df.loc[:, 'tokenized_translation'] = df.loc[:, 'translation'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
    
    # Get length of tokenized text
    df.loc[:, 'tokenized_ref_len'] = df['tokenized_reference'].apply(lambda x: len(x))
    df.loc[:, 'tokenized_trn_len'] = df['tokenized_translation'].apply(lambda x: len(x))
    return df

def remove_rows_with_length_constraints(df):
    """
    Remove rows from the DataFrame that do not meet tokenized length constraints.

    Args:
        df (pd.DataFrame): DataFrame with tokenized columns.

    Returns:
        pd.DataFrame: DataFrame after removing rows that violate length constraints.
    """
    # Remove rows with too long tokenized translation length
    df = df[df['tokenized_trn_len'] <= 75]

    # Remove samples with a very big difference between tokenized and plain lengths
    df = df[~((df['tokenized_ref_len'] > (df['reference_length'] * 1.2)) & (df['reference_length'] > 20))]
    df = df[~((df['tokenized_trn_len'] > (df['translation_length'] * 1.2)) & (df['translation_length'] > 20))]
    return df

def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with tokenized columns.

    Returns:
        pd.DataFrame: DataFrame with unnecessary columns dropped.
    """
    return df.drop(columns=['reference_length', 'translation_length', 'tokenized_ref_len', 'tokenized_trn_len', 'tokenized_reference', 'tokenized_translation'])

def save_preprocessed_data(df, output_file):
    """
    Save the preprocessed dataset to a CSV file.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        output_file (str): Path to the output CSV file.
    """
    df.to_csv(output_file)

if __name__ == "__main__":
    input_file = 'data/raw/filtered.tsv'
    output_file = 'data/internal/preprocessed.csv'
    tokenizer_model = 'SkolkovoInstitute/roberta_toxicity_classifier'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', nargs='?', const=input_file, type=str, default=input_file, help='Path to the input CSV file')
    parser.add_argument('--output_file', nargs='?', const=output_file, type=str, default=output_file, help='Path to the output preprocessed CSV file')
    parser.add_argument('--tokenizer_model', nargs='?', const=tokenizer_model, type=str, default=tokenizer_model, help='Name of the tokenizer model to use')
    args = parser.parse_args()

    df = read_and_preprocess_data(args.input_file)
    df = filter_data(df)
    df = tokenize_data(df, args.tokenizer_model)
    df = remove_rows_with_length_constraints(df)
    df = drop_unnecessary_columns(df)
    save_preprocessed_data(df, args.output_file)