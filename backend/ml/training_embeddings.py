"""Resumable frozen-encoder training features, keyed by ordered input and encoder identity."""
import hashlib
import json
import os
import numpy as np


def cached_embeddings(encoder, prompts, directory, encoder_identity):
    digest = hashlib.sha256(encoder_identity)
    for prompt in prompts:
        raw = prompt.encode('utf-8')
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    folder = directory / digest.hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    progress = folder / 'progress.json'
    target = folder / 'vectors.npy'
    completed = json.loads(progress.read_text())['completed'] if progress.exists() else 0
    vectors = np.lib.format.open_memmap(target, mode='r+' if completed else 'w+',
                                      dtype=np.float32, shape=(len(prompts), 384))
    if vectors.shape != (len(prompts), 384) or not 0 <= completed <= len(prompts):
        raise ValueError('Invalid embedding checkpoint')
    print(f'Saved training embeddings: {completed:,}/{len(prompts):,}', flush=True)
    for start in range(completed, len(prompts), 1000):
        end = min(start + 1000, len(prompts))
        vectors[start:end] = encoder.encode(prompts[start:end])
        vectors.flush()
        temporary = folder / 'progress.pending'
        temporary.write_text(json.dumps({'completed': end}))
        os.replace(temporary, progress)
        print(f'Saved training embeddings: {end:,}/{len(prompts):,}', flush=True)
    return vectors
