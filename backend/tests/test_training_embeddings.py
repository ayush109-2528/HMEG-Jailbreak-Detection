import numpy as np
from ml.training_embeddings import cached_embeddings

class Encoder:
    def __init__(self): self.calls = 0
    def encode(self, prompts):
        self.calls += 1
        return np.array([[len(p)] * 384 for p in prompts], dtype=np.float32)

def test_embedding_checkpoint_resumes_and_invalidates_changed_inputs(tmp_path):
    encoder = Encoder()
    first = cached_embeddings(encoder, ['a', 'bb'], tmp_path, b'encoder-v1')
    assert encoder.calls == 1
    again = cached_embeddings(encoder, ['a', 'bb'], tmp_path, b'encoder-v1')
    assert encoder.calls == 1
    np.testing.assert_array_equal(first, again)
    cached_embeddings(encoder, ['bb', 'a'], tmp_path, b'encoder-v1')
    cached_embeddings(encoder, ['a', 'bb'], tmp_path, b'encoder-v2')
    assert encoder.calls == 3
