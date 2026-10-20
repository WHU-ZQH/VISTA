"""Resolve bundled text annotations against an external image directory."""
import argparse
import json
from pathlib import Path


def prepare(source, image_root, output, check_images=False):
    source, image_root, output = map(lambda p: Path(p).resolve(), (source, image_root, output))
    files = sorted(source.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"No JSONL annotations found under {source}")
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the bundled annotation directory")
    plans = []
    for file in files:
        target = output / file.relative_to(source)
        if target.exists():
            raise FileExistsError(f"Use a fresh output directory: {target}")
        rows = []
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            paths = []
            for image in row["images"]:
                relative = Path(image)
                if relative.is_absolute() or relative.drive or ".." in relative.parts:
                    raise ValueError(f"Expected a relative image filename: {image}")
                # Resolve the root once; preparing text needs no per-image disk lookup.
                path = image_root / relative
                if check_images and not path.is_file():
                    raise FileNotFoundError(path)
                paths.append(str(path))
            row["images"] = paths
            rows.append(row)
        plans.append((target, rows))
    # Validate all records before writing any files.
    for target, rows in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return sum(len(rows) for _, rows in plans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    print(f"Prepared {prepare(args.source, args.image_root, args.output_dir, args.check_images)} records")
