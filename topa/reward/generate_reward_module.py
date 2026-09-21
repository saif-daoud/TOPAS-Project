"""Generate a new reward module from the template.

Usage:
  python -m topa.reward.generate_reward_module --domain my_domain
"""



import argparse
import shutil
from pathlib import Path


def _slugify(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum() or ch == "_":
            out.append(ch)
        elif ch in {" ", "-"}:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "new_domain"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a reward module from the template.")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., P4G, CBT, finance).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing domain folder if it exists.",
    )
    args = parser.parse_args()

    reward_root = Path(__file__).resolve().parent
    template_dir = reward_root / "_template"
    if not template_dir.exists():
        raise FileNotFoundError(f"Template folder not found: {template_dir}")

    slug = _slugify(args.domain)
    target_dir = reward_root / slug

    if target_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Target already exists: {target_dir}. Use --force to overwrite."
            )
        shutil.rmtree(target_dir)

    shutil.copytree(template_dir, target_dir)
    print(f"[TOPA] Created reward module: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

