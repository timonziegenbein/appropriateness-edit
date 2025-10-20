def create_llm_prompt(issue: str, sentences: str) -> str:
    """
    Formats the prompt for the LLM based on the provided issue and sentences.

    Args:
        issue: A string describing the topic of the argument.
        sentences: A string containing the sentences of the argument to be analyzed,
                   typically separated by newlines.

    Returns:
        A formatted string ready to be sent to the LLM.
    """

    prompt = f"""Task: Analyze the following argument by breaking it down into individual sentences. For each sentence, identify all inappropriate parts and edit it to make it appropriate while preserving the author's core point.

The output must be a single JSON object with a single key "sentence_edits". The value of this key should be a list of objects. Each object in the list must correspond to a sentence from the original argument and contain three keys in this specific order:
    "sentence_id": The sentence number (e.g., 1, 2, 3) corresponding to the input sentence.
    "rewritten_sentence": The full, clean, and fluent version of the rewritten sentence.
    "edits": A list of JSON objects, where each object represents a single correction and contains three keys: "inappropriate_part", "rewritten_part", and "reason". The "reason" must be one of the allowed reason values.

Definitions for inappropriateness reasons:
    Toxic Emotions: Emotions appealed to are deceptive or so intense that they discourage critical evaluation.
    Missing Commitment: The issue is not taken seriously or there is no openness to others' arguments.
    Missing Intelligibility: Meaning is unclear/irrelevant or reasoning is not understandable.
    Other Reasons: Severe orthographic errors or other issues not covered above.

Allowed Reason values:
    Toxic Emotions
    Missing Commitment
    Missing Intelligibility
    Other Reasons

Example:
    Issue: Pro choice vs pro life
    Input Sentences:
        Sentence 1: for everyone who is talking about RAPE, let me ask you one thing!!!!
        Sentence 2: if you got in a huge fight with someone and ended up breaking your hand or arm ... would you cut it off just because it would REMIND you of that expirience???
        Sentence 3: if your actualy SANE you would say no and if you say yes you need to see a Physiatrist!!!!
    JSON Output:
        {{
            "sentence_edits": [
                {{
                    "sentence_id": 1,
                    "rewritten_sentence": "For those discussing rape, consider this:",
                    "edits": [
                        {{
                            "inappropriate_part": "for everyone who is talking about",
                            "rewritten_part": "For those discussing",
                            "reason": "Missing Intelligibility"
                        }},
                        {{
                            "inappropriate_part": "RAPE",
                            "rewritten_part": "rape",
                            "reason": "Toxic Emotions"
                        }},
                        {{
                            "inappropriate_part": ", let me ask you one thing!!!!",
                            "rewritten_part": ", consider this:",
                            "reason": "Toxic Emotions"
                        }}
                    ]
                }},
                {{
                    "sentence_id": 2,
                    "rewritten_sentence": "If you got into a fight and broke your arm, would you amputate it just because it would remind you of that experience?",
                    "edits": [
                        {{
                            "inappropriate_part": "if",
                            "rewritten_part": "If",
                            "reason": "Other Reasons"
                        }},
                        {{
                            "inappropriate_part": "got in",
                            "rewritten_part": "got into",
                            "reason": "Missing Intelligibility"
                        }},
                        {{
                            "inappropriate_part": "a huge fight with someone",
                            "rewritten_part": "a fight",
                            "reason": "Toxic Emotions"
                        }},
                        {{
                            "inappropriate_part": "ended up breaking your hand or arm",
                            "rewritten_part": "broke your arm",
                            "reason": "Missing Intelligibility"
                        }},
                        {{
                            "inappropriate_part": "...",
                            "rewritten_part": ",",
                            "reason": "Other Reasons"
                        }},
                        {{
                            "inappropriate_part": "cut it off just because it would REMIND you",
                            "rewritten_part": "amputate it just because it would remind you",
                            "reason": "Missing Intelligibility"
                        }},
                        {{
                            "inappropriate_part": "expirience???",
                            "rewritten_part": "experience?",
                            "reason": "Toxic Emotions"
                        }}
                    ]
                }},
                {{
                    "sentence_id": 3,
                    "rewritten_sentence": "Most would agree that the logical response is to treat the injury, even if the memory of the event remains.",
                    "edits": [
                        {{
                            "inappropriate_part": "if your actualy SANE you would say no and if you say yes you need to see a Physiatrist!!!!",
                            "rewritten_part": "Most would agree that the logical response is to treat the injury, even if the memory of the event remains.",
                            "reason": "Toxic Emotions"
                        }}
                    ]
                }}
            ]
        }}

Now complete the task for the following:

Issue: {issue}
Input Sentences: {sentences}
JSON Output:\n"""

    return " ".join(prompt.replace("\n", " ").replace("\t", " ").split())
    #return prompt
