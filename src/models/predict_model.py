from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

# Function to set up the model and tokenizer
def setup_model_and_tokenizer(model_name, checkpoint_name, max_length):
    """
    Initialize the model and tokenizer for inference.

    Args:
        model_name (str): Name of the pre-trained model.
        checkpoint_name (str): Name of the model checkpoint.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        model: Initialized sequence-to-sequence model.
        tokenizer: Initialized tokenizer.
    """
    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_length, cache_dir=f".cache/tokenizers/{model_name}")

    # Load inference model
    model = AutoModelForSeq2SeqLM.from_pretrained(f'models/{model_name}/{checkpoint_name}')
    model.eval()
    model.config.use_cache = False

    return model, tokenizer

# Function to perform model inference on text
def translate(model, inference_request, tokenizer, max_length):
    """
    Translate input text using the model.

    Args:
        model: Initialized sequence-to-sequence model.
        inference_request (str): Input text for translation.
        tokenizer: Initialized tokenizer.
        max_length (int): Maximum sequence length for tokenization.

    Returns:
        str: Translated text.
    """
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)

# Main function for user interaction
def main():
    """
    Main function for user interaction.
    """
    
    checkpoint_name = "best"
    max_length = 85
    model_name = 'SkolkovoInstitute/bart-base-detox'

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", nargs='?', const=max_length, type=int, default=max_length, help='Maximum length of tokens for inference model to generate')
    parser.add_argument("--checkpoint_name", nargs='?', const=checkpoint_name, type=str, default=checkpoint_name, help='Name of the checkpoint to get results from')
    parser.add_argument('--model_name', nargs='?', const=model_name, type=str, default=model_name, help='Name of the model to use')

    args = parser.parse_args()

    model, tokenizer = setup_model_and_tokenizer(args.model_name, args.checkpoint_name, args.max_length)

    inference_request = input("Please input your inference request (press enter to exit):\n")
    try:
        while inference_request != '':
            # Perform translation
            translation = translate(model, inference_request, tokenizer, args.max_length)
            print(translation + '\n')

            inference_request = input("Please input your inference request (press enter to exit):\n")
    finally:
        print('Exiting...')

if __name__ == "__main__":
    main()
