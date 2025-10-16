import argparse
import torch
import torch.nn as nn
import numpy as np
import random

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import numpy as np
import pandas as pd
import math
import difflib
import sys
sys.path.append('/mnt/home/tziegenb/appropriateness-feedback/src/end-to-end')

from utils.model_defs import LanguageModel, PositionalEncoding
from utils.reward_functions import (
    load_reward_models,
    calculate_perplexity_for_sequence,
    compute_fluency_scores
)
from sentence_transformers import util

def find_sentence_window(text, start_char, end_char):
    # Simple regex to find sentence boundaries
    sentence_boundaries = [m.end() for m in re.finditer(r'[.!?]', text)]
    
    window_start = 0
    for i in range(len(sentence_boundaries) - 1, -1, -1):
        if sentence_boundaries[i] < start_char:
            window_start = sentence_boundaries[i] + 1
            break
            
    window_end = len(text)
    for i in range(len(sentence_boundaries)):
        if sentence_boundaries[i] >= end_char:
            window_end = sentence_boundaries[i]
            break
            
    return window_start, window_end

# --- Sequence Generation Logic ---
def process_example_difflib(reconstructed_argument, inappropriate_part, rewritten_part, tokenizer):
    encoding = tokenizer(reconstructed_argument, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    offsets = encoding['offset_mapping']

    if not tokens:
        return None

    tags = ['keep'] * len(tokens)
    start_char = reconstructed_argument.find(inappropriate_part)
    if start_char == -1:
        return None
    end_char = start_char + len(inappropriate_part)

    token_start_index = -1
    token_end_index = -1
    for i, offset in enumerate(offsets):
        token_start, token_end = offset
        if start_char < token_end and end_char > token_start:
            if token_start_index == -1:
                token_start_index = i
            token_end_index = i
    
    if token_start_index != -1:
        before_edit_tokens = tokens[token_start_index:token_end_index+1]
        
        if not isinstance(rewritten_part, str):
            rewritten_part = ""
        after_edit_tokens = tokenizer.tokenize(rewritten_part)

        edit_tags = []
        matcher = difflib.SequenceMatcher(None, before_edit_tokens, after_edit_tokens)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                edit_tags.extend(['keep'] * (i2 - i1))
            elif tag == 'delete':
                edit_tags.extend(['del'] * (i2 - i1))
            elif tag == 'replace':
                edit_tags.extend(['replace'] * (i2 - i1))
            elif tag == 'insert':
                edit_tags.extend(['add'] * (j2 - j1))
        
        if len(edit_tags) == (token_end_index - token_start_index + 1):
            tags[token_start_index:token_end_index+1] = edit_tags
    return tags

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Test OOD detection for a language model.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output file for the results.')
    parser.add_argument('--percentile-threshold', type=float, default=1.5131214332580567, help='The percentile threshold for OOD detection.')
    parser.add_argument('--ss-threshold', type=float, default=0.7572, help='The semantic similarity threshold for OOD detection.')
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ss_model, _, _, _, fluency_model, fluency_tokenizer = load_reward_models(args.model_path, device)

    vocab = {'<pad>': 0, 'keep': 1, 'del': 2, 'add': 3, 'replace': 4}
    process_example = process_example_difflib

    embedding_dim = 200
    nhead = 2
    nhid = 200
    nlayers = 2
    max_len = 500
    dropout = 0.2
    
    model = LanguageModel(len(vocab), embedding_dim, nhead, nhid, nlayers, dropout).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # --- Test Examples ---
    examples = [
        "<Missing Intelligibility>for everyone who is talking about→For those discussing</Missing Intelligibility> <Toxic Emotions>RAPE→rape</Toxic Emotions><Other Reasons> →, </Other Reasons> <Toxic Emotions>let me ask you one thing!!!!→I would like to pose another scenario.</Toxic Emotions> <Other Reasons>if→If</Other Reasons> you got <Missing Intelligibility>in→into</Missing Intelligibility> a <Toxic Emotions>huge fight with someone→fight</Toxic Emotions> and <Missing Intelligibility>ended up breaking→broke your</Missing Intelligibility> hand or arm <Other Reasons>...→,</Other Reasons> would you <Missing Intelligibility>cut it off just because it would REMIND you→amputate it to remove the physical reminder</Missing Intelligibility> <Other Reasons>expirience???→experience?</Other Reasons> <Toxic Emotions>if your actualy SANE you would say no→Of course, the answer is no.</Toxic Emotions> <Missing Commitment>and if you say yes you need to see a Physiatrist!!!!→Although it is understandable to have emotional scars, physical scars can be treated and lived with, even if they remain as a reminder of the experience.</Missing Commitment>",
        "<Toxic Emotions>You really think criminals, who already have unregisterted guns, want other citizens to have them?\nIf I was a criminal, and I used a gun, I would want guns to be controlled so that there is a far, far, far less chance of being shot while i rob/rape/murder→Criminals who possess illegal guns pose a significant threat to society. The debate over gun control is a complex issue that involves various perspectives. On one hand, some argue that gun control measures would deprive law-abiding citizens of their right to self-defense, while others claim that such measures would help reduce the number of gun-related crimes. To examine this issue, let's consider the following points. Firstly, studies haveshown that countries with strict gun control laws tend to have lower rates of gun-related deaths. Secondly, gun control measures can help prevent guns from falling into the wrong hands, which could potentially lead to a decrease in gun-related crimes. However, it's also important to note that gun control measures must be carefully crafted to avoid infringing on the rights of law-abiding citizens. For instance, measures such as background checks and red flag laws can help ensure that guns are not being sold to individuals who are prohibited from possessing them. Ultimately, the question of whether guns should be controlled is a matter of debate, and there is no easy answer. Perhaps we can work together to find a solution that balances the need for public safety with the need to protect the rights of law-abiding citizens.</Toxic Emotions>",
        "<Missing Commitment>I can not beleive that you guys are standing up for a man who talks down your troops.→I have concerns that some people may be defending a commentator who has expressed opinions that could be perceived as disrespecting members of the military.</Missing Commitment>",
        "<Missing Intelligibility>That's seems to me→That seems to me</Missing Intelligibility> like a <Toxic Emotions>perfect faith based→perfectly faith-based</Toxic Emotions> answer.\n<Missing Intelligibility>And I think I have suitably→And I think I have sufficiently</Missing Intelligibility> answered an <Other Reasons>Aethiests→atheists' </Other Reasons> <Missing Intelligibility>retroactive and logical conclusion→retroactive, logical conclusion</Missing Intelligibility>.\nThe <Missing Intelligibility>answer in both cases then is, No.→This answer, however, still doesn't address the original paradox: No.</Missing Intelligibility>",
        "<Toxic Emotions>And yet you continue to use your elementary-school level insults→The use of insults, particularly those of an elementary-school level, is unproductive</Toxic Emotions> instead of debating using facts and arguments. <Other Reasons>Pathetic.→Generally, rhetorical appeals to emotion do little to sway a discussion.</Other Reasons>",
        "<Other Reasons>Oh and there→Oh, and there</Other Reasons> <Toxic Emotions>was also→also</Toxic Emotions> <Other Reasons>constant→ongoing</Other Reasons> <Missing Intelligibility>terrorist attacks by Jewish fundamentalist groups against Britain and the Palestinians prior to and during the early days of the state of Israel. They also attacked the U.S. in Egypt.→The argument that George Bush caused 9/11 is unrelated to prior terrorist attacks against Britain and the Palestinians or attacks in Egypt. The relevance of these events to the question of whether George Bush caused 9/11 is unclear.</Missing Intelligibility>"
    ]

    with open(args.output_file, "w") as f:
        f.write(f"Human-like Threshold: {args.percentile_threshold or 'N/A'}\n")
        f.write(f"Semantic Similarity Threshold: {args.ss_threshold}\n")

        for i, example in enumerate(examples):
            f.write(f"--- Example {i+1} ---\n")
            reconstructed_argument = re.sub(r"<([^>]+)>(.*?)→(.*?)</\1>", r"\2", example, flags=re.DOTALL)
            f.write(f"Reconstructed Argument: {reconstructed_argument}\n\n")
            
            edits = re.findall(r"<([^>]+)>(.*?)→(.*?)</\1>", example, flags=re.DOTALL)
            if not edits:
                f.write("No edits found.")
                continue

            for _, inappropriate_part, rewritten_part in edits:
                f.write(f"Inappropriate part: {inappropriate_part}\n")
                f.write(f"Rewritten part: {rewritten_part}\n")

                sequence = process_example(reconstructed_argument, inappropriate_part, rewritten_part, tokenizer)
                
                if sequence:
                    perplexity = calculate_perplexity_for_sequence(sequence, model, vocab, device, max_len)
                    f.write(f"Human-like Perplexity: {perplexity:.4f}\n")

                    if args.percentile_threshold is not None:
                        if perplexity > args.percentile_threshold:
                            f.write("Human-like Result: Out-of-Distribution (Not human-like)\n")
                        else:
                            f.write("Human-like Result: In-Distribution (Human-like)\n")
                    else:
                        f.write("Human-like Result: N/A (no threshold provided)\n")

                    # Semantic Similarity Check
                    if inappropriate_part and rewritten_part:
                        before_text = reconstructed_argument
                        after_text = reconstructed_argument.replace(inappropriate_part, rewritten_part, 1)
                        embedding1 = ss_model.encode(before_text, convert_to_tensor=True)
                        embedding2 = ss_model.encode(after_text, convert_to_tensor=True)
                        ss_score = util.pytorch_cos_sim(embedding1, embedding2).item()
                        f.write(f"Semantic Similarity: {ss_score:.4f}\n")
                        if ss_score >= args.ss_threshold:
                            f.write("Semantic Similarity Result: Similar\n")
                        else:
                            f.write("Semantic Similarity Result: Dissimilar\n")

                    # Fluency Score Check
                    if fluency_model and fluency_tokenizer:
                        start_char = reconstructed_argument.find(inappropriate_part)
                        end_char = start_char + len(inappropriate_part)

                        window_start, window_end = find_sentence_window(reconstructed_argument, start_char, end_char)
                        original_sentence = reconstructed_argument[window_start:window_end]
                        modified_sentence = original_sentence.replace(inappropriate_part, rewritten_part, 1)

                        fluency_scores = compute_fluency_scores([original_sentence, modified_sentence])
                        if fluency_scores and len(fluency_scores) > 0:
                            fluency_score = fluency_scores[0]
                            if fluency_score > 0.0:
                                f.write(f"Fluency Result: ACCEPTABLE (Score: {fluency_score:.4f})\n")
                            else:
                                f.write(f"Fluency Result: HARMFUL (Score: {fluency_score:.4f})\n")

                else:
                    f.write(f"Could not process example for inappropriate part: {inappropriate_part}\n")
                f.write("\n")

if __name__ == "__main__":
    main()
