import pandas as pd
import numpy as np
import math

def main():
    try:
        df = pd.read_csv('results/semantic_similarity_classification.csv')
        scores = df['semantic_similarity'].tolist()
        
        mean = np.mean(scores)
        std = np.std(scores)
        n = len(scores)
        
        ci = 1.96 * (std / math.sqrt(n))
        
        print(f"Mean: {mean}")
        print(f"Std dev: {std}")
        print(f"95% Confidence Interval: ({mean - ci}, {mean + ci})")
        percentile_1 = np.percentile(scores, 1)
        percentile_5 = np.percentile(scores, 5)
        print(f"1st percentile of semantic similarity: {percentile_1}")
        print(f"5th percentile of semantic similarity: {percentile_5}")

    except FileNotFoundError:
        print("Error: The file results/ss_scores.csv was not found.")
    except KeyError:
        print("Error: The CSV file must have a column named 'ss_score'.")

if __name__ == '__main__':
    main()
