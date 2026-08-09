from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kvcache_service.config import Settings
from kvcache_service.domain import BuildCacheCommand, CompletionCommand
from kvcache_service.errors import CacheCompatibilityError
from kvcache_service.store import SafeTensorCacheStore
from kvcache_service.transformers_backend import TransformersBackend

from .fakes import FakeCausalLM, FakeTokenizer


class TransformersBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        settings = Settings(
            model_id="fake/model",
            model_revision="test-commit",
            device="cpu",
            store_dir=Path(self.temporary.name),
            verify_checksum=True,
            max_context_tokens=128,
            default_chunk_size=2,
            max_new_tokens=16,
        )
        self.backend = TransformersBackend(
            settings,
            model=FakeCausalLM(self.torch),
            tokenizer=FakeTokenizer(),
            store=SafeTensorCacheStore(Path(self.temporary.name)),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_chunked_prefill_persists_and_reuses_cache(self) -> None:
        info = self.backend.build_cache(
            BuildCacheCommand(text="abcdef", input_ids=None, chunk_size=2)
        )
        self.assertEqual(info.token_count, 7)  # BOS plus six input tokens.
        self.assertEqual(info.layer_count, 1)
        tensor_path = Path(self.temporary.name) / info.cache_id / "tensors.safetensors"
        self.assertTrue(tensor_path.is_file())

        result = self.backend.complete(
            CompletionCommand(
                cache_id=info.cache_id,
                prompt="g",
                input_ids=None,
                max_tokens=3,
                temperature=0,
                top_p=1,
                top_k=0,
                seed=7,
                stop=[],
                stop_token_ids=[],
            )
        )
        self.assertEqual(result.cached_tokens, info.token_count)
        self.assertEqual(result.prompt_tokens, info.token_count + 1)
        self.assertEqual(len(result.token_ids), 3)
        self.assertGreaterEqual(result.timings_ms["total"], result.timings_ms["decode"])

    def test_build_is_content_addressed_and_idempotent(self) -> None:
        command = BuildCacheCommand(text="same", input_ids=None, chunk_size=2)
        first = self.backend.build_cache(command)
        second = self.backend.build_cache(command)
        self.assertEqual(first.cache_id, second.cache_id)
        self.assertEqual(len(self.backend.list_caches()), 1)

    def test_full_prompt_mode_verifies_and_removes_cached_prefix(self) -> None:
        info = self.backend.build_cache(BuildCacheCommand(text="abc", input_ids=None, chunk_size=2))
        result = self.backend.complete(
            CompletionCommand(
                cache_id=info.cache_id,
                prompt="abcXYZ",
                input_ids=None,
                max_tokens=2,
                temperature=0,
                top_p=1,
                top_k=0,
                seed=7,
                stop=[],
                stop_token_ids=[],
                prompt_mode="full",
            )
        )
        self.assertEqual(result.prompt_tokens, info.token_count + 3)

        with self.assertRaises(CacheCompatibilityError):
            self.backend.complete(
                CompletionCommand(
                    cache_id=info.cache_id,
                    prompt="xbcXYZ",
                    input_ids=None,
                    max_tokens=2,
                    temperature=0,
                    top_p=1,
                    top_k=0,
                    seed=7,
                    stop=[],
                    stop_token_ids=[],
                    prompt_mode="full",
                )
            )

    def test_can_generate_directly_from_prefix_without_suffix(self) -> None:
        info = self.backend.build_cache(BuildCacheCommand(text="abc", input_ids=None, chunk_size=2))
        result = self.backend.complete(
            CompletionCommand(
                cache_id=info.cache_id,
                prompt=None,
                input_ids=None,
                max_tokens=1,
                temperature=0,
                top_p=1,
                top_k=0,
                seed=None,
                stop=[],
                stop_token_ids=[],
            )
        )
        self.assertEqual(result.prompt_tokens, info.token_count)

    def test_explicit_model_fingerprint_invalidates_old_cache(self) -> None:
        first_settings = Settings(
            model_id="fake/model",
            model_revision="mutable",
            model_fingerprint="release-a",
            device="cpu",
            store_dir=Path(self.temporary.name),
            max_context_tokens=128,
        )
        second_settings = Settings(
            model_id="fake/model",
            model_revision="mutable",
            model_fingerprint="release-b",
            device="cpu",
            store_dir=Path(self.temporary.name),
            max_context_tokens=128,
        )
        first_backend = TransformersBackend(
            first_settings, model=FakeCausalLM(self.torch), tokenizer=FakeTokenizer()
        )
        second_backend = TransformersBackend(
            second_settings, model=FakeCausalLM(self.torch), tokenizer=FakeTokenizer()
        )
        command = BuildCacheCommand(text="same", input_ids=None, chunk_size=2)
        first = first_backend.build_cache(command)
        second = second_backend.build_cache(command)
        self.assertNotEqual(first.cache_id, second.cache_id)


if __name__ == "__main__":
    unittest.main()
