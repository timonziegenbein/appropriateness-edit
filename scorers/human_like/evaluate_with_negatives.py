"""
Comprehensive evaluation of human-like scorer using positive and negative examples.

Generates:
- Perplexity distribution plots
- ROC curve and AUC
- Precision-Recall curve
- Threshold recommendations
"""

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from typing import Tuple, List

from scorers.human_like.model_defs import LanguageModel


def calculate_perplexity_for_sequence(sequence: str, model: nn.Module, vocab: dict,
                                     device: torch.device, max_len: int) -> float:
    """Calculate perplexity for a given sequence."""
    sequence_as_int = [vocab.get(token, 0) for token in sequence.split(',')]

    if len(sequence_as_int) <= 1:
        return float('inf')

    input_seq = sequence_as_int[:-1]
    target_seq = sequence_as_int[1:]

    # Pad sequences
    padded_input = np.array(
        input_seq[:max_len] + [vocab['<pad>']] * (max_len - len(input_seq))
        if len(input_seq) < max_len else input_seq[:max_len]
    )
    padded_target = np.array(
        target_seq[:max_len] + [vocab['<pad>']] * (max_len - len(target_seq))
        if len(target_seq) < max_len else target_seq[:max_len]
    )

    inputs = torch.from_numpy(padded_input).long().unsqueeze(0).to(device)
    targets = torch.from_numpy(padded_target).long().unsqueeze(0).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    with torch.no_grad():
        output = model(inputs)
        loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
        perplexity = torch.exp(loss)
        return perplexity.item()


def compute_perplexities(df: pd.DataFrame, model: nn.Module, vocab: dict,
                        device: torch.device, max_len: int) -> pd.DataFrame:
    """Compute perplexity scores for all sequences in DataFrame."""
    print("Computing perplexity scores...")
    perplexities = []

    for idx, row in df.iterrows():
        try:
            ppl = calculate_perplexity_for_sequence(
                row['sequence'], model, vocab, device, max_len
            )
            perplexities.append(ppl)
        except Exception as e:
            print(f"Warning: Failed to compute perplexity for row {idx}: {e}")
            perplexities.append(float('inf'))

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} sequences...")

    df['perplexity'] = perplexities
    # Filter out infinite perplexities
    df = df[df['perplexity'] != float('inf')]

    return df


def plot_perplexity_distributions(df: pd.DataFrame, output_path: str):
    """Plot perplexity distributions for positive and negative examples."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    pos_ppl = df[df['label'] == 1]['perplexity']
    neg_ppl = df[df['label'] == 0]['perplexity']

    axes[0].hist(pos_ppl, bins=50, alpha=0.5, label='Positive (Human)', color='green', density=True)
    axes[0].hist(neg_ppl, bins=50, alpha=0.5, label='Negative (Synthetic)', color='red', density=True)
    axes[0].set_xlabel('Perplexity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Perplexity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Box plot
    df_plot = df[['perplexity', 'label']].copy()
    df_plot['label'] = df_plot['label'].map({1: 'Positive', 0: 'Negative'})
    sns.boxplot(x='label', y='perplexity', data=df_plot, ax=axes[1])
    axes[1].set_title('Perplexity Box Plot')
    axes[1].set_ylabel('Perplexity')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved perplexity distribution plot to {output_path}")
    plt.close()


def plot_roc_curve(df: pd.DataFrame, output_path: str) -> Tuple[float, float]:
    """Plot ROC curve and return AUC and optimal threshold."""
    # Convert perplexity to binary predictions (lower perplexity = human-like = positive)
    # So we need to negate perplexity for ROC computation
    y_true = df['label'].values
    y_scores = -df['perplexity'].values  # Negate so higher score = more human-like

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (maximizes TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]  # Convert back to perplexity

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
               label=f'Optimal (ppl={optimal_threshold:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved ROC curve to {output_path}")
    plt.close()

    return roc_auc, optimal_threshold


def plot_precision_recall_curve(df: pd.DataFrame, output_path: str) -> Tuple[float, float, float]:
    """Plot Precision-Recall curve and return AP, optimal threshold, and F1."""
    y_true = df['label'].values
    y_scores = -df['perplexity'].values  # Negate so higher score = more human-like

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap_score = average_precision_score(y_true, y_scores)

    # Find threshold that maximizes F1 score
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = -thresholds[optimal_idx]  # Convert back to perplexity
    optimal_f1 = f1_scores[optimal_idx]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
    plt.scatter(recall[optimal_idx], precision[optimal_idx], marker='o', color='red', s=100,
               label=f'Optimal F1={optimal_f1:.3f} (ppl={optimal_threshold:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Precision-Recall curve to {output_path}")
    plt.close()

    return ap_score, optimal_threshold, optimal_f1


def compute_threshold_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Compute classification metrics for a given threshold."""
    y_true = df['label'].values
    y_pred = (df['perplexity'] <= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'threshold': threshold,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate human-like scorer with positive and negative examples.'
    )
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--test-csv', type=str, required=True,
                       help='Path to test CSV with positive and negative examples')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--vocab-type', type=str, default='v2',
                       choices=['v1', 'v2'],
                       help='Vocabulary type (v1: 5 tokens, v2: 6 tokens with keep-in-edit)')
    parser.add_argument('--max-len', type=int, default=500,
                       help='Maximum sequence length')

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Vocabulary
    if args.vocab_type == 'v1':
        vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4}
    else:  # v2
        vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4, 'keep-in-edit': 5}

    print(f"Using vocabulary: {vocab}")

    # Load model
    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2

    model = LanguageModel(len(vocab), embedding_dim, nhead, nhid, nlayers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Load test data
    df = pd.read_csv(args.test_csv)
    print(f"Loaded {len(df)} test examples ({df['label'].sum()} positive, {(1-df['label']).sum()} negative)")

    # Compute perplexities
    df = compute_perplexities(df, model, vocab, device, args.max_len)
    print(f"Successfully computed perplexity for {len(df)} examples")

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Statistics
    print("\n=== Perplexity Statistics ===")
    print("\nPositive examples (human edits):")
    print(df[df['label'] == 1]['perplexity'].describe())
    print("\nNegative examples (synthetic):")
    print(df[df['label'] == 0]['perplexity'].describe())

    # Plot distributions
    plot_perplexity_distributions(df, f"{args.output_dir}/perplexity_distributions.png")

    # ROC curve
    print("\n=== ROC Analysis ===")
    roc_auc, roc_optimal_threshold = plot_roc_curve(df, f"{args.output_dir}/roc_curve.png")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"ROC optimal threshold (perplexity): {roc_optimal_threshold:.4f}")

    # Precision-Recall curve
    print("\n=== Precision-Recall Analysis ===")
    ap_score, pr_optimal_threshold, optimal_f1 = plot_precision_recall_curve(
        df, f"{args.output_dir}/precision_recall_curve.png"
    )
    print(f"Average Precision: {ap_score:.4f}")
    print(f"PR optimal threshold (perplexity): {pr_optimal_threshold:.4f}")
    print(f"Optimal F1 Score: {optimal_f1:.4f}")

    # Compute metrics at different thresholds
    print("\n=== Threshold Analysis ===")

    # Percentile-based thresholds
    pos_ppl = df[df['label'] == 1]['perplexity']
    percentile_thresholds = [
        ('50th percentile', np.percentile(pos_ppl, 50)),
        ('75th percentile', np.percentile(pos_ppl, 75)),
        ('90th percentile', np.percentile(pos_ppl, 90)),
        ('95th percentile', np.percentile(pos_ppl, 95)),
        ('99th percentile', np.percentile(pos_ppl, 99)),
    ]

    thresholds_to_test = [
        ('ROC Optimal', roc_optimal_threshold),
        ('PR Optimal (Max F1)', pr_optimal_threshold),
    ] + percentile_thresholds

    results = []
    for name, threshold in thresholds_to_test:
        metrics = compute_threshold_metrics(df, threshold)
        metrics['name'] = name
        results.append(metrics)
        print(f"\n{name} (threshold={threshold:.4f}):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f}")
        print(f"  FNR:       {metrics['fnr']:.4f}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{args.output_dir}/threshold_metrics.csv", index=False)
    print(f"\nSaved threshold metrics to {args.output_dir}/threshold_metrics.csv")

    # Save perplexity scores
    df.to_csv(f"{args.output_dir}/perplexity_scores.csv", index=False)
    print(f"Saved perplexity scores to {args.output_dir}/perplexity_scores.csv")

    print("\n=== Evaluation Complete ===")
    print(f"All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
