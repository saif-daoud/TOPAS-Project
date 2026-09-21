import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

@dataclass(frozen=True)
class CacheKey:
    name: str
    fingerprint: str

    def filename(self) -> str:
        return f"{self.name}__{self.fingerprint}.pkl"

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def fingerprint_dict(d: Dict[str, Any]) -> str:
    """Stable-ish fingerprint for cache invalidation.

    Only include small metadata: paths, sizes, params. Do NOT include full data payloads.
    """
    items = []
    for k in sorted(d.keys()):
        v = d[k]
        items.append((k, str(v)))
    blob = repr(items).encode("utf-8")
    return _sha1_bytes(blob)

def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)

def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)

def cache_or_build(cache_dir: Path, key: CacheKey, build_fn: Callable[[], Any], force_rebuild: bool = False) -> Any:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / key.filename()
    if (not force_rebuild) and cache_path.exists():
        return load_pickle(cache_path)
    obj = build_fn()
    save_pickle(obj, cache_path)
    return obj