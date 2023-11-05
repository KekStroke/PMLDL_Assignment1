# Baselines: 
## Copying the initial text
My first idea was to just copy the initial text and calculate metrics for it.
## Deletion of toxic words
For the second baseline solution I have deleted all toxic words from all text according to [this list of toxic words](https://github.com/s-nlp/detox/blob/main/emnlp2021/style_transfer/condBERT/vocab/toxic_words.txt).
# Hypothesis 1: Custom encoder-decoder based model
My initial idea was to check whether some one have implemented encoder-decoder based models to solve any machine translation task. I have build my own toxic and nontoxic vocabulary based on the train dataset and have trained the [very basic model](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html). 
# Hypothesis 2: Pretrained tokenizer
My second hypothesis was to find our if there is any other already existing vocabulary for the task because previous model was successfully leaning of the training set but very inconsistently on the test set. So I have though than a more general vocabulary will help. [Vocabulary was taken from here](https://huggingface.co/s-nlp/roberta_toxicity_classifier)
# Hypothesis 3: Pretrained BART model
Previous model has shown some success but it was definitely not good enough. Since I don't have much resources I have decided to fine tune one model that have already been trained for text detoxification task. I have found a [BERT paradetox model](https://github.com/s-nlp/paradetox) to fine-tune.
# Results
|Method   |STA↓   |Sim↑   |FL↑   |BLEU↑  |
|---|:---:|:---:|:---:|:---:|
|**Baselines**|
|Duplicate   |1.00   |0.80   |1.00   |1.00|
|Delete   |0.34   |0.77   |0.91   |0.83|
|**Encoder-decoder based**|
|Custom vocab   |**0.00**   |0.12   |0.21   |0.00   |
|Pretrained tokenizer   |**0.00**   |0.72   |0.35   |0.26   |
|**BARD Fine-tuning**|
|BARD base detox   |**0.00**   |**0.84**   |**0.92**   |**0.38**   |