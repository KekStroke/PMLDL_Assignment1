import torch
import matplotlib.pyplot as plt
import argparse

def show_plot(val_scores):
    """
    Display a plot of validation scores.

    Args:
        val_scores (list): List of validation scores to be plotted.
    """
    plt.figure()
    plt.plot(val_scores, label='BLUE score')
    plt.legend(loc="best")
    plt.title(f'Scores on epoch {len(val_scores)}')
    plt.show()

def main(args):
    """
    Main function for loading a checkpoint and displaying validation scores.

    Args:
        args (argparse.Namespace): Command line arguments parsed using argparse.
    """
    load_ckpt_path = f'models/{args.model_name}'
    model_ckpt_path = load_ckpt_path + '/' + args.checkpoint_name
    ckpt_path = model_ckpt_path + '_checkpoint.pt'

    checkpoint = torch.load(ckpt_path)
    val_scores = checkpoint['val_scores']

    show_plot(val_scores)

if __name__ == "__main__":
    checkpoint_name = 'best'
    model_name = 'SkolkovoInstitute/bart-base-detox'

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", nargs='?', const=checkpoint_name, type=str, default=checkpoint_name, help='Name of the checkpoint to get results from')
    parser.add_argument('--model_name', nargs='?', const=model_name, type=str, default=model_name, help='Name of the model to use')

    args = parser.parse_args()

    main(args)
