import torch
import matplotlib.pyplot as plt
import argparse

def showPlot(val_scores):
    plt.figure()
    plt.plot(val_scores, label='BLUE score')
    plt.legend(loc="best")
    plt.title(f'Scores on epoch {len(val_scores)}')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name")
    parser.add_argument("--model_name")
    args = parser.parse_args()

    checkpoint_name = 'best'
    model_name = 'SkolkovoInstitute/bart-base-detox'
    
    if (args.checkpoint_name is not None): 
        checkpoint_name = args.checkpoint_name
    if (args.model_name is not None): 
        model_name = args.model_name
        

    load_ckpt_path = f'models/{model_name}'
    
    model_ckpt_path = load_ckpt_path+'/' + checkpoint_name
    ckpt_path = model_ckpt_path+'_checkpoint.pt'

    checkpoint = torch.load(ckpt_path)
    val_scores = checkpoint['val_scores']

    showPlot(val_scores)
