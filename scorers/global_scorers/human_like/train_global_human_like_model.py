import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import json
import difflib
import numpy as np
from pathlib import Path
import wandb
import weave
from transformers import AutoTokenizer
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scorers.local_scorers.human_like.model_defs import LanguageModel

# V2 vocab with 'keep-in-edit' token
hl_vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}


class GlobalEditSequenceDataset(Dataset):
    """Dataset for document-level edit sequences from IteraTeR."""

    def __init__(self, json_file, vocab, max_len, tokenizer):
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sequences = []
        self.sequence_metadata = []  # Track metadata for analysis

        print(f"Loading document-level edit sequences from {json_file}...")
        with open(json_file, 'r') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    before_revision = data.get('before_revision', '')
                    edit_actions = data.get('edit_actions', [])

                    if not before_revision or not edit_actions:
                        continue

                    # Generate edit sequence
                    edit_sequence, metadata = self._generate_edit_sequence_with_metadata(before_revision, edit_actions)

                    if edit_sequence and len(edit_sequence) > 1:
                        self.sequences.append(edit_sequence)
                        self.sequence_metadata.append(metadata)

                        # Log sample sequences to Weave periodically
                        if len(self.sequences) % 1000 == 0:
                            self._log_sample_to_weave(i, before_revision, edit_sequence, metadata)

                    if (i + 1) % 1000 == 0:
                        print(f"  Processed {i + 1} documents, collected {len(self.sequences)} sequences")

                except Exception as e:
                    continue

        print(f"Loaded {len(self.sequences)} document-level edit sequences")

        # Log dataset statistics to Weave
        self._log_dataset_statistics()

    @weave.op(tracing_sample_rate=0.01)
    def _generate_edit_sequence_with_metadata(self, before_revision, edit_actions):
        """Generate document-level edit sequence from edit actions with metadata tracking."""
        # Tokenize the original document
        encoding = self.tokenizer(before_revision, return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        offsets = encoding['offset_mapping']

        metadata = {
            'num_tokens': len(tokens),
            'num_edit_actions': len([e for e in edit_actions if e.get('type') in ['R', 'A', 'D']]),
            'text_length': len(before_revision),
            'edit_type_counts': {'R': 0, 'A': 0, 'D': 0},
            'operations_applied': 0,
        }

        if len(tokens) == 0:
            return [], metadata

        # Initialize all tokens as 'keep'
        tags = ['keep'] * len(tokens)

        # Process each edit action
        for edit in edit_actions:
            edit_type = edit.get('type')
            start_char = edit.get('start_char_pos')
            end_char = edit.get('end_char_pos')

            if start_char is None or edit_type not in ['R', 'A', 'D']:
                continue

            metadata['edit_type_counts'][edit_type] += 1

            # Find token indices that overlap with this edit
            token_start_index = -1
            token_end_index = -1
            for i, offset in enumerate(offsets):
                token_start, token_end = offset
                if start_char < token_end and (end_char is None or end_char > token_start):
                    if token_start_index == -1:
                        token_start_index = i
                    token_end_index = i

            if token_start_index == -1:
                continue

            metadata['operations_applied'] += 1

            # Apply the edit operation to the tag sequence
            if edit_type == 'R':  # Replace
                after_text = edit.get('after', '')
                before_tokens = tokens[token_start_index:token_end_index+1]
                after_tokens = self.tokenizer.tokenize(after_text) if after_text else []

                # Use difflib to get fine-grained operations
                matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens)
                current_idx = token_start_index
                for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                    if tag == 'equal':
                        # Tokens that match within an edit region are 'keep-in-edit'
                        for idx in range(current_idx, current_idx + (i2 - i1)):
                            if idx < len(tags):
                                tags[idx] = 'keep-in-edit'
                        current_idx += (i2 - i1)
                    elif tag == 'delete':
                        for idx in range(current_idx, current_idx + (i2 - i1)):
                            if idx < len(tags):
                                tags[idx] = 'del'
                        current_idx += (i2 - i1)
                    elif tag == 'replace':
                        for idx in range(current_idx, current_idx + (i2 - i1)):
                            if idx < len(tags):
                                tags[idx] = 'replace'
                        current_idx += (i2 - i1)
                    elif tag == 'insert':
                        for _ in range(j2 - j1):
                            tags.insert(current_idx, 'add')
                            current_idx += 1

            elif edit_type == 'D':  # Delete
                for idx in range(token_start_index, token_end_index + 1):
                    if idx < len(tags):
                        tags[idx] = 'del'

            elif edit_type == 'A':  # Add
                after_text = edit.get('after', '')
                after_tokens = self.tokenizer.tokenize(after_text) if after_text else []
                for _ in range(len(after_tokens)):
                    tags.insert(token_start_index, 'add')

        # Add sequence statistics to metadata
        metadata['sequence_length'] = len(tags)
        metadata['tag_counts'] = {
            'keep': tags.count('keep'),
            'keep-in-edit': tags.count('keep-in-edit'),
            'del': tags.count('del'),
            'add': tags.count('add'),
            'replace': tags.count('replace'),
        }
        # Edit ratio: proportion of tokens that were modified (excluding 'keep' and 'keep-in-edit')
        unchanged_tokens = metadata['tag_counts']['keep'] + metadata['tag_counts']['keep-in-edit']
        metadata['edit_ratio'] = 1.0 - (unchanged_tokens / max(1, len(tags)))

        return tags, metadata

    @weave.op()
    def _log_sample_to_weave(self, doc_index, before_revision, edit_sequence, metadata):
        """Log a sample edit sequence to Weave for inspection."""
        # Truncate text for readability
        text_preview = before_revision[:200] + "..." if len(before_revision) > 200 else before_revision
        sequence_preview = edit_sequence[:100] + ['...'] if len(edit_sequence) > 100 else edit_sequence

        weave.publish({
            'sample_type': 'global_edit_sequence',
            'doc_index': doc_index,
            'text_preview': text_preview,
            'sequence_preview': sequence_preview,
            'metadata': metadata,
        })

    @weave.op()
    def _log_dataset_statistics(self):
        """Log overall dataset statistics to Weave."""
        if not self.sequence_metadata:
            return

        # Aggregate statistics
        total_sequences = len(self.sequence_metadata)
        avg_sequence_length = np.mean([m['sequence_length'] for m in self.sequence_metadata])
        avg_num_edits = np.mean([m['num_edit_actions'] for m in self.sequence_metadata])
        avg_edit_ratio = np.mean([m['edit_ratio'] for m in self.sequence_metadata])

        # Count edit types
        total_edit_type_counts = {'R': 0, 'A': 0, 'D': 0}
        for m in self.sequence_metadata:
            for edit_type, count in m['edit_type_counts'].items():
                total_edit_type_counts[edit_type] += count

        # Sequence length distribution
        sequence_lengths = [m['sequence_length'] for m in self.sequence_metadata]

        stats = {
            'dataset_statistics': {
                'total_sequences': total_sequences,
                'avg_sequence_length': float(avg_sequence_length),
                'avg_num_edits': float(avg_num_edits),
                'avg_edit_ratio': float(avg_edit_ratio),
                'total_edit_type_counts': total_edit_type_counts,
                'sequence_length_percentiles': {
                    'p10': float(np.percentile(sequence_lengths, 10)),
                    'p25': float(np.percentile(sequence_lengths, 25)),
                    'p50': float(np.percentile(sequence_lengths, 50)),
                    'p75': float(np.percentile(sequence_lengths, 75)),
                    'p90': float(np.percentile(sequence_lengths, 90)),
                    'p99': float(np.percentile(sequence_lengths, 99)),
                    'max': float(np.max(sequence_lengths)),
                }
            }
        }

        weave.publish(stats)
        print(f"\nDataset Statistics:")
        print(f"  Total sequences: {total_sequences}")
        print(f"  Avg sequence length: {avg_sequence_length:.1f}")
        print(f"  Avg num edits: {avg_num_edits:.1f}")
        print(f"  Avg edit ratio: {avg_edit_ratio:.3f}")
        print(f"  Edit type distribution: {total_edit_type_counts}")
        print(f"  Sequence length percentiles: {stats['dataset_statistics']['sequence_length_percentiles']}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence_as_int = [self.vocab.get(token, 0) for token in sequence]

        # Create input and target sequences
        input_seq = sequence_as_int[:-1]
        target_seq = sequence_as_int[1:]

        # Pad or truncate to max_len
        padded_input = input_seq[:self.max_len] + [self.vocab['<pad>']] * max(0, self.max_len - len(input_seq))
        padded_target = target_seq[:self.max_len] + [self.vocab['<pad>']] * max(0, self.max_len - len(target_seq))

        return torch.tensor(padded_input), torch.tensor(padded_target)


def main():
    parser = argparse.ArgumentParser(description='Train a global language model on document-level edit sequences.')
    parser.add_argument('--input-json', type=str, default='datasets/scorers/IteraTeR/full_doc_level/train.json',
                       help='Path to the IteraTeR JSON file.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--max-len', type=int, default=1500, help='Maximum sequence length (longer for documents).')

    # Wandb/Weave configuration
    parser.add_argument('--wandb-project', type=str, default='global-human-like-scorer',
                       help='Weights & Biases project name.')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run.')

    args = parser.parse_args()

    # --- Initialize Wandb and Weave ---
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_len': args.max_len,
            'embedding_dim': 200,
            'nhead': 2,
            'nhid': 200,
            'nlayers': 2,
            'dropout': 0.2,
        }
    )

    # Initialize Weave for tracing
    weave.init(project_name=args.wandb_project)

    print(f"Vocabulary size: {len(hl_vocab)}")
    print(f"Vocabulary: {hl_vocab}")
    wandb.config.update({'vocab_size': len(hl_vocab)})

    # --- Dataset and DataLoader ---
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    print("Loading dataset...")
    dataset = GlobalEditSequenceDataset(args.input_json, hl_vocab, args.max_len, tokenizer)
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
    model = LanguageModel(len(hl_vocab), embedding_dim, nhead, nhid, nlayers, dropout).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    wandb.config.update({'model_parameters': num_params})

    # --- Training ---
    criterion = nn.CrossEntropyLoss(ignore_index=hl_vocab['<pad>'])
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

            loss = criterion(output.view(-1, len(hl_vocab)), targets.view(-1))
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

            if (i+1) % 50 == 0:
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': hl_vocab,
        'max_len': args.max_len,
    }, args.model_path)
    print(f"Model saved to {args.model_path}")

    # Save model as wandb artifact
    artifact = wandb.Artifact(
        name=f"global-human-like-model",
        type="model",
        description="Global human-like edit scorer language model for document-level patterns"
    )
    artifact.add_file(args.model_path)
    wandb.log_artifact(artifact)

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
