import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('/mnt/home/tziegenb/appropriateness-feedback/src/end-to-end')
from utils.model_defs import LanguageModel, EditSequenceDataset


def calculate_perplexity(model, data_loader, device, vocab):
    model.eval()
    perplexities = []
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.long().to(device), targets.long().to(device)
            output = model(inputs)
            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            perplexity = torch.exp(loss)
            perplexities.append(perplexity.item())

    return perplexities

def main():
    parser = argparse.ArgumentParser(description='Calculate perplexity of edit sequences.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--input-csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to the output CSV file.')
    args = parser.parse_args()

    # --- Hyperparameters ---
    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2
    max_len = 500
    dropout = 0.2

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Vocabulary ---
    vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4}

    # --- Load Model ---
    model = LanguageModel(len(vocab), embedding_dim, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(args.model_path))

    # --- Create Dataset and DataLoader ---
    dataset = EditSequenceDataset(args.input_csv, vocab, max_len)
    data_loader = DataLoader(dataset, batch_size=1) # Batch size 1 for individual perplexity

    # --- Calculate Perplexity ---
    perplexities = calculate_perplexity(model, data_loader, device, vocab)

    # --- Save Perplexity Scores ---
    perplexity_df = pd.DataFrame(perplexities, columns=['perplexity'])
    perplexity_df.to_csv(args.output_csv, index=False)
    print(f"Perplexity scores saved to {args.output_csv}")

    # --- Calculate and Print Statistics ---
    mean_perplexity = np.mean(perplexities)
    std_perplexity = np.std(perplexities)
    percentile_99 = np.percentile(perplexities, 99)
    confidence_interval = 1.96 * (std_perplexity / np.sqrt(len(perplexities)))
    print(f"Mean perplexity: {mean_perplexity}")
    print(f"Standard deviation of perplexity: {std_perplexity}")
    print(f"99th percentile of perplexity: {percentile_99}")
    print(f"95% confidence interval: ({mean_perplexity - confidence_interval}, {mean_perplexity + confidence_interval})")

if __name__ == '__main__':
    main()
