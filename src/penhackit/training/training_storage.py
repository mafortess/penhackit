from pathlib import Path


def list_dataset_candidates(datasets_dir: Path) -> list[Path]:
    """
    Devuelve una lista de 'dataset_dir' candidatos.
    - Añade cada subdirectorio que contenga dataset.jsonl.
    - Añade el propio datasets_dir si contiene dataset.jsonl (caso plano).
    Sort by mtime (últimos primero).
    """
    candidates = []

    # caso plano: datasets/dataset.jsonl
    if (datasets_dir / "dataset.jsonl").exists():
        candidates.append(datasets_dir)

    # caso normal: datasets/<name>/dataset.jsonl
    if datasets_dir.exists():
        for p in datasets_dir.iterdir():
            if p.is_dir() and (p / "dataset.jsonl").exists():
                candidates.append(p)

    # orden por mtime (últimos primero)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates

