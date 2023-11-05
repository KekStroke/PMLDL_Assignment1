Author: Anvar Iskhakov\
Email: an.iskhakov@innopolis.university\
Group: BS21-AI

# Prerequisites

### Dependencies
Install all required packages with
```bash
pip install -r requirements.txt
```

### Model weights
Download [models folder](https://drive.google.com/drive/folders/1r81s9v-OvfWYgB6-xJtNKw4TH0DrVvBZ?usp=sharing) and pasted it into `/models` folder in the project root

### Raw data
Download [raw dataset folder](https://drive.google.com/drive/folders/1uQwi-MRTmdok_xjbl3WLJYJ-UnYPGOze?usp=sharing) and pasted it into `/data/raw` folder in the project root

# Prepare Data
To pre-process dataset to further training enter following command for the repository root:
```bash
python src/data/make_dataset.py 
```
One can add some arguments as well. Command with default arguments is:
```bash
python src/data/make_dataset.py --input_file data/raw/filtered.tsv --output_file data/internal/preprocessed.csv --tokenizer_model SkolkovoInstitute/roberta_toxicity_classifier 
```

# Train model
To train final model on the preprocessed dataset enter following command for the repository root:
```bash
python src/models/train_model.py 
```
One can add some arguments as well. Command with default arguments is:
```bash
$ python src/models/train_model.py --max_length 85 --batch_size 64 --checkpoint_name test --model_name SkolkovoInstitute/bart-base-detox --train_ratio 0.8 -
-val_test_ratio 0.5 --learning_rate 0.00002 --weight_decay 0.01 --save_total_limit 1 --num_train_epochs 1 --save_steps 500 --eval_steps 500 --logging_steps 100

```


# Inference
To use the final trained model on your own sentences enter following command for the repository root:
```bash
python src/models/predict_model.py
```
One can add some arguments as well. Command with default arguments is:
```bash
python src/models/predict_model.py --checkpoint_name best --max_length 85 --model_name SkolkovoInstitute/bart-base-detox
```

# Miscellaneous
Other hypotheses were tested in notebooks that can be found in `/notebooks/extra_hypotheses`