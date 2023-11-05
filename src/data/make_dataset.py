import sys
import pandas as pd
from transformers import RobertaTokenizer

# read dataset and Remove unnecessary column
df = pd.read_csv('data/raw/filtered.tsv', sep='\t', index_col=0)
df = df.drop('lenght_diff', axis=1)

# Swap columns with ref_tox and trn_tox mixed up
df_mixed_up = df[(df['ref_tox'] < df['trn_tox'])]
df_mixed_up.columns = ['translation', 'reference', 'similarity', 'trn_tox', 'ref_tox']
df_proper = df
df_proper.loc[df_mixed_up.index] = df_mixed_up.loc[df_mixed_up.index]

# Create ref len and trn len columns
df_proper['reference_length'] = df_proper['reference'].apply(lambda x: len(x.split()))
df_proper['translation_length'] = df_proper['translation'].apply(lambda x: len(x.split()))

# Get only rows with very toxic reference and with very low toxicity translation
df_similar = df_proper[(df_proper['ref_tox'] > 0.99) & (df_proper['trn_tox'] < 0.01) & (df_proper['similarity'] > 0.7) & (df_proper['reference_length'] <= 60) & (df_proper['translation_length'] <= 60)]

df = df_similar

# Get tokenizer trained on toxic words to get tokens of words
tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier', cache_dir=".cache/tokenizers/roberta_toxicity_classifier")

df.loc[:, 'tokenized_reference'] = df.loc[:, 'reference'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
df.loc[:, 'tokenized_translation'] = df.loc[:, 'translation'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Get length of tokenized text
df.loc[:, 'tokenized_ref_len'] = df['tokenized_reference'].apply(lambda x: len(x))
df.loc[:, 'tokenized_trn_len'] = df['tokenized_translation'].apply(lambda x: len(x))

# Remove rows samples with too long tokenized translation length
df = df[df['tokenized_trn_len'] <= 75]

# Remove samples with very big difference between tokenized and plain lengths
df = df[~((df['tokenized_ref_len'] > (df['reference_length'] * 1.2)) & (df['reference_length'] > 20))]
df= df[~((df['tokenized_trn_len'] > (df['translation_length'] * 1.2)) & (df['translation_length'] > 20))]

# Drop unnecessary columns
df = df.drop(columns=['reference_length', 'translation_length', 'tokenized_ref_len', 'tokenized_trn_len', 'tokenized_reference', 'tokenized_translation'])

# Save preprocessed dataset
df.to_csv('data/internal/preprocessed.csv')