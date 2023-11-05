from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from transformers import pipeline, TextClassificationPipeline
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import evaluate

def blue(preds, refs):
    assert len(refs) == len(preds)
    
    sacrebleu = evaluate.load("sacrebleu", cache_dir="../.cache/metrics/sacrebleu")
    
    results = sacrebleu.compute(predictions=preds, references=refs)
        
    return results['score']

def content_similarity(args, preds, refs):
    assert len(refs) == len(preds)
    results = torch.zeros(len(preds))
    cos = torch.nn.CosineSimilarity(dim=1).to(device=args['device'])
    
    model = SentenceTransformer('sentence-transformers/LaBSE', device=args['device'], cache_folder='../.cache/models')
    
    try:
        with torch.no_grad():
            for i in tqdm(range(0, len(preds), args['batch_size'])):
                ref_embeddings = model.encode(refs[i:i + args['batch_size']], convert_to_tensor=True, device=args['device'])
                pred_embeddings = model.encode(preds[i:i + args['batch_size']], convert_to_tensor=True, device=args['device'])
            
                results[i:i + args['batch_size']] = cos(ref_embeddings, pred_embeddings)
    except Exception as e:
        raise e
    finally:
        del model
        del cos
        
    return results.mean().item()

def fluency(args, preds):
    # load tokenizer and model weights
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/roberta-large-cola-krishna2020", cache_dir="../.cache/tokenizers/roberta-large-cola-krishna2020")    
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/roberta-large-cola-krishna2020", cache_dir="../.cache/models/roberta-large-cola-krishna2020").to(device=args['device'])

    class MyPipeline(TextClassificationPipeline):
     def postprocess(self, model_outputs):
         best_class = model_outputs["logits"][0].softmax(dim=-1)
         return best_class[0]

    classifier = pipeline("text-classification", model=model, batch_size=args['batch_size'], tokenizer=tokenizer, device=args['device'], pipeline_class=MyPipeline)
    
    predictions = torch.tensor(classifier(preds))
    
    del model
    del tokenizer
    
    return predictions.mean().item()

def style_transfer_accuracy(args, preds):
    print('Calculating style of predictions')
    results = torch.zeros(len(preds))
    
    # load tokenizer and model weights
    tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier', cache_dir="../.cache/tokenizers/roberta_toxicity_classifier")
    model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier', cache_dir="../.cache/models/roberta_toxicity_classifier").to(device=args['device'])

    class MyPipeline(TextClassificationPipeline):
     def postprocess(self, model_outputs):
         return model_outputs["logits"][0][1]

    classifier = pipeline("text-classification", model=model, batch_size=args['batch_size'], tokenizer=tokenizer, device=args['device'], pipeline_class=MyPipeline)
    
    logits = torch.tensor(classifier(preds))
    
    results = torch.softmax(logits, -1)
    
    del model
    del tokenizer
        
    return results.mean().item()