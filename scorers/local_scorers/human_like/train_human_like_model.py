import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append('/mnt/home/tziegenb/appropriateness-feedback/src/end-to-end')

from utils.model_defs import LanguageModel, EditSequenceDataset


def main():
    parser = argparse.ArgumentParser(description='Train a language model on edit sequences.')
    parser.add_argument('--input-csv', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    args = parser.parse_args()

    # --- Vocabulary ---
    vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4}
    print(f"Vocabulary size: {len(vocab)}")

    # --- Dataset and DataLoader ---
    max_len = 500
    dataset = EditSequenceDataset(args.input_csv, vocab, max_len)
    print(f"Loaded {len(dataset)} sequences.")
    train_loader = DataLoader(dataset, shuffle=True, batch_size=64)

    # --- Model and Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2
    dropout = 0.2
    model = LanguageModel(len(vocab), embedding_dim, nhead, nhid, nlayers, dropout).to(device)

    # --- Training ---
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.long().to(device), targets.long().to(device)
            optimizer.zero_grad()
            output = model(inputs)
            
            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/5], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f"Epoch {epoch+1} finished.")

    print("Training finished.")
    # --- Save Model ---
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
