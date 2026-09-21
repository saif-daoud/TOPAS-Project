import argparse
from pathlib import Path
import pandas as pd
from data_loading import DataRoot, load_conv_state_annotations
from utils import set_seed, save_json

def split_users_greedy(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Split by *users* (~80/20) while keeping session proportion close."""
    sess_counts = df.groupby("user_idx")["session_idx"].nunique().to_dict()
    users = sorted(sess_counts.keys())

    total_sessions = sum(sess_counts.values())
    target_val_sessions = total_sessions * val_frac
    target_k = max(1, int(round(len(users) * val_frac)))

    set_seed(seed)

    remaining = users[:]
    # Start from high-session users so we can match the session proportion.
    remaining.sort(key=lambda u: (-sess_counts[u], u))

    val_users = []
    val_sessions = 0

    for _ in range(target_k):
        best_u = None
        best_diff = None
        for u in remaining:
            diff = abs((val_sessions + sess_counts[u]) - target_val_sessions)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_u = u
        val_users.append(best_u)
        val_sessions += sess_counts[best_u]
        remaining.remove(best_u)

    train_users = sorted([u for u in users if u not in set(val_users)])
    val_users = sorted(val_users)

    return train_users, val_users, sess_counts, total_sessions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="encoders_RL root (has data/, annotations/, components/)")
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    root = DataRoot(Path(args.root))
    df = load_conv_state_annotations(root)

    train_users, val_users, sess_counts, total_sessions = split_users_greedy(df, val_frac=args.val_frac, seed=args.seed)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (root.root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split = {
        "train_user_idxs": train_users,
        "val_user_idxs": val_users,
        "sessions_per_user": {str(k): int(v) for k, v in sess_counts.items()},
        "total_sessions": int(total_sessions),
        "val_sessions": int(sum(sess_counts[u] for u in val_users)),
        "train_sessions": int(sum(sess_counts[u] for u in train_users)),
        "val_frac_requested": float(args.val_frac),
        "val_frac_actual_users": float(len(val_users) / len(sess_counts)),
        "val_frac_actual_sessions": float(sum(sess_counts[u] for u in val_users) / total_sessions),
    }

    save_json(split, out_dir / "user_split.json")
    print("Saved:", out_dir / "user_split.json")
    print("Users (train/val):", len(train_users), len(val_users))
    print("Sessions (train/val):", split["train_sessions"], split["val_sessions"])
    print("val_frac_actual_users=", split["val_frac_actual_users"], "val_frac_actual_sessions=", split["val_frac_actual_sessions"])

if __name__ == "__main__":
    main()
