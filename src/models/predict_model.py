from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

checkpoint_name = "best"
MAX_LENGTH = 85

# Set model name
model_name = 'SkolkovoInstitute/bart-base-detox'

# Get tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=MAX_LENGTH, cache_dir=f".cache/tokenizers/{model_name}")

# Load inference model
model = AutoModelForSeq2SeqLM.from_pretrained(f'models/{model_name}/{checkpoint_name}')
model.eval()
model.config.use_cache = False

# Function to perform model inference on some text
def translate(model, inference_request, tokenizer=tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_new_tokens=MAX_LENGTH)
    return tokenizer.decode(outputs[0], skip_special_tokens=True,temperature=0)

inference_request = input("Please input your inference request (press enter to exit):\n")
try:
    while inference_request != '':
        # Perform translation
        translation = translate(model, inference_request,tokenizer)

        print(translation+'\n')
        
        inference_request = input("Please input your inference request (press enter to exit):\n")
finally:
    print('Exiting...')