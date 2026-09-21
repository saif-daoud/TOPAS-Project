"""Extract target-LLM internal representations for each system turn."""

import argparse
import re
from pathlib import Path
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..activations import short_model_name


def parse_dtype(name: str):
    name = str(name).strip().lower()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    return "auto"


def parse_layers(raw: str) -> Optional[list[int]]:
    raw = str(raw or "").strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def extract_session_activations(
    model,
    tokenizer,
    session_text: str,
    system_tag: str,
    user_tag: str,
    select_layers: Optional[list[int]] = None,
) -> dict:
    model.eval()

    inputs = tokenizer(session_text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"].to(model.device)

    system_token_id = tokenizer.convert_tokens_to_ids(system_tag)
    assert system_token_id != tokenizer.unk_token_id
    turn_positions = torch.nonzero(input_ids == system_token_id, as_tuple=True)[0]
    model_turn_positions = turn_positions.to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0).to(model.device),
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    activations = torch.stack([layer_hidden[0][model_turn_positions] for layer_hidden in outputs.hidden_states])
    layers = list(range(len(outputs.hidden_states)))

    if select_layers is not None:
        activations = activations[select_layers]
        layers = list(select_layers)

    return {
        "activations": activations.detach().cpu(),
        "turn_token_positions": turn_positions.detach().cpu(),
        "assistant_positions": turn_positions.detach().cpu(),
        "layers": layers,
        "indexing": "system_turn_idx",
        "system_tag": system_tag,
        "system_token_id": int(system_token_id),
        "user_tag": user_tag,
        "model_name": getattr(model.config, "name_or_path", "unknown"),
        "num_system_turns": int(turn_positions.numel()),
    }


def sanity_check(save_dict: dict, session_text: str, system_tag: str, out_path: Path) -> None:
    activations = save_dict["activations"]
    expected_turns = session_text.count(system_tag)
    extracted_turns = int(activations.shape[1])
    position_count = int(save_dict["turn_token_positions"].numel())

    assert activations.ndim == 3
    assert extracted_turns > 0
    assert expected_turns == extracted_turns == position_count
    assert int(activations.shape[0]) == len(save_dict["layers"])

    print(f"[sanity] {out_path}: shape={tuple(activations.shape)} turns={extracted_turns}")


def session_files(transcripts_dir: Path) -> list[tuple[str, int, int, Path]]:
    out = []
    id_dirs = sorted(
        [p for p in transcripts_dir.iterdir() if p.is_dir() and re.match(r"^[A-Za-z][A-Za-z0-9._-]*_\d+$", p.name)],
        key=lambda p: p.name,
    )
    for id_dir in id_dirs:
        prefix, user_idx = id_dir.name.rsplit("_", 1)
        for path in sorted(id_dir.glob("session_*.txt"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])):
            session_idx = int(path.stem.rsplit("_", 1)[-1])
            out.append((prefix, int(user_idx), session_idx, path))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--transcripts_dir", type=str, default="extract_internal_representation/transcripts")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--model_id", type=str, required=True)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--system_tag", type=str, required=True)
    ap.add_argument("--user_tag", type=str, required=True)
    ap.add_argument("--select_layers", type=str, default="")
    ap.add_argument("--store_session_text", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    transcripts_dir = Path(args.transcripts_dir)
    if not transcripts_dir.is_absolute():
        transcripts_dir = (root / transcripts_dir).resolve()

    out_dir = Path(args.out_dir) if args.out_dir else Path("extract_internal_representation") / f"activations_{short_model_name(args.model_id)}"
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [args.system_tag, args.user_tag]})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map=args.device_map,
        torch_dtype=parse_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    select_layers = parse_layers(args.select_layers)
    total = 0

    files = session_files(transcripts_dir)
    assert files

    for prefix, user_idx, session_idx, path in tqdm(files, desc="sessions"):
        session_text = path.read_text(encoding="utf-8")
        save_dict = extract_session_activations(
            model,
            tokenizer,
            session_text,
            system_tag=args.system_tag,
            user_tag=args.user_tag,
            select_layers=select_layers,
        )
        if args.store_session_text:
            save_dict["session_text"] = session_text

        out_user_dir = out_dir / f"{prefix}_{user_idx}"
        out_user_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_user_dir / f"session_{session_idx}.pt"
        sanity_check(save_dict, session_text, args.system_tag, out_path)
        torch.save(save_dict, out_path)
        total += 1

    print(f"Saved {total} activation files to: {out_dir}")


if __name__ == "__main__":
    main()
