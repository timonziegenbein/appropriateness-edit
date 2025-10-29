import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
from pathlib import Path
import wandb
import weave
import pandas as pd
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scorers.local_scorers.human_like.model_defs import LanguageModel, EditSequenceDataset


def main():
    parser = argparse.ArgumentParser(description='Train a language model on edit sequences (v2 with keep-in-edit token).')
    parser.add_argument('--input-csv', type=str, default=None,
                       help='Path to the input CSV file (alternative to --dataset-name).')
    parser.add_argument('--dataset-name', type=str, default='timonziegenbein/human-like-edit-sequences',
                       help='HuggingFace dataset name (alternative to --input-csv).')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use for training (default: train).')
    parser.add_argument('--model-path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')

    # Wandb/Weave configuration
    parser.add_argument('--wandb-project', type=str, default='human-like-scorer',
                       help='Weights & Biases project name (also used for Weave tracing).')
    parser.add_argument('--wandb-run-id', type=str, default=None,
                       help='Weights & Biases run ID to resume from.')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run.')

    args = parser.parse_args()

    # --- Initialize Wandb and Weave ---
    wandb.init(
        project=args.wandb_project,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        name=args.run_name,
        config={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_len': 500,
            'embedding_dim': 200,
            'nhead': 2,
            'nhid': 200,
            'nlayers': 2,
            'dropout': 0.2,
        }
    )

    # Initialize Weave for tracing
    weave.init(project_name=args.wandb_project)

    # --- Updated Vocabulary with keep-in-edit token ---
    vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Vocabulary: {vocab}")
    wandb.config.update({'vocab_size': len(vocab)})

    # --- Dataset and DataLoader ---
    max_len = 500

    # Load dataset from CSV or HuggingFace
    if args.input_csv:
        print(f"Loading dataset from CSV: {args.input_csv}")
        dataset = EditSequenceDataset(args.input_csv, vocab, max_len)
        wandb.config.update({'dataset_source': args.input_csv})
    else:
        print(f"Loading dataset from HuggingFace: {args.dataset_name}, split: {args.split}")
        hf_dataset = load_dataset(args.dataset_name, split=args.split)

        # Convert HuggingFace dataset to CSV temporarily for EditSequenceDataset
        temp_csv = '/tmp/human_like_sequences_train.csv'
        df = pd.DataFrame({
            'sequence': hf_dataset['sequence'],
            'label': hf_dataset['label']
        })
        df.to_csv(temp_csv, index=False)

        dataset = EditSequenceDataset(temp_csv, vocab, max_len)
        wandb.config.update({
            'dataset_source': args.dataset_name,
            'dataset_split': args.split
        })

    print(f"Loaded {len(dataset)} sequences.")
    wandb.config.update({'dataset_size': len(dataset)})
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)

    # --- Model and Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2
    dropout = 0.2
    model = LanguageModel(len(vocab), embedding_dim, nhead, nhid, nlayers, dropout).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    wandb.config.update({'model_parameters': num_params})

    # --- Training ---
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        total_loss = 0
        epoch_losses = []

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.long().to(device), targets.long().to(device)
            optimizer.zero_grad()
            output = model(inputs)

            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            epoch_losses.append(loss_item)

            # Calculate perplexity
            perplexity = torch.exp(loss).item()

            # Log to wandb every step
            wandb.log({
                'train/loss': loss_item,
                'train/perplexity': perplexity,
                'train/epoch': epoch + 1,
                'train/step': global_step,
            }, step=global_step)

            global_step += 1

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss_item:.4f}, Perplexity: {perplexity:.4f}')

        avg_loss = total_loss / len(train_loader)
        avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

        # Log epoch summary
        wandb.log({
            'train/epoch_avg_loss': avg_loss,
            'train/epoch_avg_perplexity': avg_perplexity,
            'epoch': epoch + 1,
        }, step=global_step)

        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}, Average Perplexity: {avg_perplexity:.4f}")

    print("Training finished.")

    # --- Save Model ---
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

    # Save model as wandb artifact
    artifact = wandb.Artifact(
        name=f"human-like-model-v2",
        type="model",
        description="Human-like edit scorer language model v2 with keep-in-edit token"
    )
    artifact.add_file(args.model_path)
    wandb.log_artifact(artifact)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
