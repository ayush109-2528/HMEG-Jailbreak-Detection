"""Download the pinned official CPU encoder at build/setup time, never during requests."""
import hashlib
import json
import urllib.request
from ml.experts.semantic_encoder import REVISION, asset_dir


def main():
    folder = asset_dir()
    folder.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REVISION}/"
    files = {"model.onnx": "onnx/model_quint8_avx2.onnx", "tokenizer.json": "tokenizer.json"}
    manifest = {"repository": "sentence-transformers/all-MiniLM-L6-v2", "revision": REVISION, "max_tokens": 256, "pooling": "masked mean, L2 normalized", "files": {}}
    for name, remote in files.items():
        target = folder / name
        pending = folder / (name + ".download")
        urllib.request.urlretrieve(base + remote, pending)
        pending.replace(target)
        manifest["files"][name] = {"source": remote, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size}
        print("Downloaded", name, target.stat().st_size, "bytes", flush=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
