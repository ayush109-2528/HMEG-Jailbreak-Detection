"""Pinned MiniLM ONNX inference with masked mean pooling and L2 normalization."""
import os
from pathlib import Path
from threading import Lock
import numpy as np

REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

def asset_dir():
    return Path(os.getenv("SEMANTIC_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "semantic")))

class SemanticEncoder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        folder = asset_dir()
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(str(folder / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.lock = Lock()

    def encode(self, prompts, show_progress_bar=False):
        vectors = []
        with self.lock:
            for start in range(0, len(prompts), 8):
                batch = self.tokenizer.encode_batch(list(prompts[start:start + 8]))
                inputs = {"input_ids": np.array([x.ids for x in batch], dtype=np.int64),
                          "attention_mask": np.array([x.attention_mask for x in batch], dtype=np.int64),
                          "token_type_ids": np.array([x.type_ids for x in batch], dtype=np.int64)}
                inputs = {x.name: inputs[x.name] for x in self.session.get_inputs()}
                hidden = self.session.run(None, inputs)[0]
                mask = inputs["attention_mask"][..., None].astype(np.float32)
                pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
                pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
                vectors.append(pooled)
        return np.concatenate(vectors) if vectors else np.empty((0, 384), dtype=np.float32)
