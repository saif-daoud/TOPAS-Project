import json
from collections import Counter
from math import ceil
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

def _collapse_consecutive(seq: List[str]) -> List[str]:
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out

def _load_macro_sequences(macro_csv_path: str, collapse_consecutive: bool = True) -> List[List[str]]:
    df = pd.read_csv(macro_csv_path)
    df = df[df["selected_macro_action"].notna()].copy()
    df = df.sort_values(["user_idx", "session_idx", "from_idx"]) # phases

    MACRO_KEYS = {
        "Alliance": "Therapeutic Alliance and Engagement",
        "Agenda": "Agenda, Goal Setting, and Action Planning",
        "Assessment": "Assessment, Diagnosis, and Case Formulation",
        "Restructuring": "Cognitive Interventions: Identification and Restructuring",
        "SkillsTraining": "Skills Generalization, Homework, and Progress Review",
        "CoreBeliefs": "Core Beliefs and Schema Work",
        "Psychoeducation": "Psychoeducation and Therapy Orientation",
        "Activation": "Behavioral Activation, Experimentation, and Skills Training",
        "Monitoring": "Ongoing Progress Monitoring and Troubleshooting",
        "CrisisManagement": "Crisis Management and Safety Planning",
        "RelapsePrevention": "Relapse Prevention and Termination Planning",
        "Exposure": "Exposure and Response Prevention",
    }

    macro = df["selected_macro_action"].unique().tolist()
    # print(set(macro) - set(MACRO_KEYS.values()))
    intersection = list(set(macro) - set(MACRO_KEYS.values()))
    # Use short names for the macro actions
    if len(intersection) == 1 and intersection[0] == "uncertain":
        df["selected_macro_action"] = df['selected_macro_action'].map({x: idx for idx, x in MACRO_KEYS.items()})

    sequences = []
    for (_, _), g in df.groupby(["user_idx", "session_idx"], sort=False):
        seq = [str(x) for x in g["selected_macro_action"].tolist()]
        # remove uncertain phases from sequence mining
        seq = [x for x in seq if x.lower() != "uncertain"]
        if collapse_consecutive:
            seq = _collapse_consecutive(seq)
        if len(seq) > 0:
            sequences.append(seq)
    return sequences

def _frequent_contiguous_ngrams(sequences: List[List[str]], min_support: int, min_pattern_length: int, 
                                max_pattern_length: int) -> List[Tuple[int, List[str]]]:
    """Support = number of sessions containing the pattern at least once.
    Patterns are contiguous n-grams (a simple, dependency-free approximation).
    """
    support = Counter()

    for seq in sequences:
        seen = set()
        L = len(seq)
        for n in range(min_pattern_length, max_pattern_length + 1):
            for i in range(0, L - n + 1):
                pat = tuple(seq[i : i + n])
                seen.add(pat)
        for pat in seen:
            support[pat] += 1

    patterns = [(sup, list(pat)) for pat, sup in support.items() if sup >= min_support]
    patterns = sorted(patterns, key=lambda x: (-x[0], -len(x[1]), x[1]))
    return patterns

def mine_frequent_macro_patterns(macro_csv_path: str, out_dir: str, min_support_ratio: float = 0.5, 
                                 min_pattern_length: int = 2, max_pattern_length: int = 10, top_k: int = 30, 
                                 collapse_consecutive: bool = True) -> str:
    """Mine frequent macro-action sequences and save results next to annotations.

    - Primary method: PrefixSpan (sub-sequences, not necessarily contiguous)
    - Fallback: contiguous n-grams (no extra dependency)

    Outputs:
      - frequent_macro_patterns.json
      - frequent_macro_patterns.csv
      - frequent_macro_patterns.png (bar plot)
    """
    out_dir = str(Path(out_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sequences = _load_macro_sequences(macro_csv_path, collapse_consecutive=collapse_consecutive)

    min_support = max(1, ceil(len(sequences) * float(min_support_ratio)))
    patterns = _frequent_contiguous_ngrams(sequences, min_support=min_support, min_pattern_length=min_pattern_length, 
                                           max_pattern_length=max_pattern_length)
    patterns = patterns[: int(top_k)]

    payload = {
        "macro_csv_path": str(macro_csv_path),
        "num_sessions": len(sequences),
        "min_support_ratio": float(min_support_ratio),
        "min_support": int(min_support),
        "min_pattern_length": int(min_pattern_length),
        "max_pattern_length": int(max_pattern_length),
        "top_k": int(top_k),
        "patterns": [
            {
                "support": int(sup),
                "pattern": pat,
                "pattern_str": " -> ".join(pat),
            }
            for sup, pat in patterns
        ],
    }

    json_path = str(Path(out_dir) / "frequent_macro_patterns.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # CSV
    df = pd.DataFrame(
        {
            "support": [p[0] for p in patterns],
            "length": [len(p[1]) for p in patterns],
            "pattern": [" -> ".join(p[1]) for p in patterns],
        }
    )
    csv_path = str(Path(out_dir) / "frequent_macro_patterns.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    plt.figure(figsize=(12, max(4, 0.25 * len(df))))
    plt.barh(range(len(df)), df["support"].tolist())
    plt.yticks(range(len(df)), df["pattern"].tolist())
    plt.gca().invert_yaxis()
    plt.xlabel("Support (#sessions)")
    plt.title("Frequent Macro-Action Patterns")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "frequent_macro_patterns.pdf"), dpi=200, bbox_inches="tight")
    plt.close()

    return json_path