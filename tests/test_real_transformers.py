from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from kvcache_service.config import Settings
from kvcache_service.domain import BuildCacheCommand, CompletionCommand
from kvcache_service.transformers_backend import TransformersBackend

from .fakes import FakeTokenizer


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "local inference dependencies are not installed",
)
class RealTransformersCompatibilityTest(unittest.TestCase):
    def test_random_tiny_gpt2_cache_round_trip(self) -> None:
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=32,
            n_positions=64,
            n_embd=16,
            n_layer=2,
            n_head=2,
            eos_token_id=None,
        )
        model = GPT2LMHeadModel(config)
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(
                model_id="random/tiny-gpt2",
                model_revision="test",
                device="cpu",
                store_dir=Path(directory),
                max_context_tokens=64,
                default_chunk_size=3,
                max_new_tokens=8,
            )
            backend = TransformersBackend(settings, model=model, tokenizer=FakeTokenizer())
            info = backend.build_cache(
                BuildCacheCommand(text="abcdefgh", input_ids=None, chunk_size=3)
            )
            result = backend.complete(
                CompletionCommand(
                    cache_id=info.cache_id,
                    prompt="question",
                    input_ids=None,
                    max_tokens=2,
                    temperature=0,
                    top_p=1,
                    top_k=0,
                    seed=1,
                    stop=[],
                    stop_token_ids=[],
                )
            )
            self.assertEqual(info.layer_count, 2)
            self.assertEqual(result.cached_tokens, info.token_count)
            self.assertEqual(len(result.token_ids), 2)
