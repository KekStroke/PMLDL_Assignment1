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
# Train model
To train final model on the preprocessed dataset enter following command for the repository root:
```bash
python src/models/train_model.py 
```
# Inference
To use the final trained model on your own sentences enter following command for the repository root:
```bash
python src/models/predict_model.py
```