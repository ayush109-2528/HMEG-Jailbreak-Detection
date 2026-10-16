"""Integration checks use official encoder assets when installed; no research scores invented."""
import numpy as np
import pytest
from ml.experts.semantic_encoder import SemanticEncoder, asset_dir

@pytest.fixture(scope="module")
def encoder():
    if not (asset_dir() / "model.onnx").exists():
        pytest.skip("Run scripts.download_semantic to test the real encoder")
    return SemanticEncoder()

def test_embedding_batch_padding_and_normalization(encoder):
    short = "A rainbow forms in sunlight."
    single = encoder.encode([short])[0]
    batch = encoder.encode([short, "Explain the optical phenomena behind sunlight passing through raindrops."])
    assert batch.shape == (2, 384)
    assert np.isfinite(batch).all()
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-5)
    # Dynamic quantization can change slightly when padding changes batch ranges.
    assert float(single @ batch[0]) > .99

def test_long_prompt_is_bounded_and_deterministic(encoder):
    text = "science " * 3000
    first = encoder.encode([text])
    second = encoder.encode([text])
    assert first.shape == (1, 384)
    np.testing.assert_allclose(first, second, atol=1e-6)
    assert encoder.encode([]).shape == (0, 384)
