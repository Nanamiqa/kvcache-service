from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from kvcache_service.app import create_app
from kvcache_service.config import Settings

from .fakes import FakeBackend


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(Settings(), FakeBackend()))

    def test_cache_lifecycle_and_completion(self) -> None:
        created = self.client.post("/v1/kv-caches", json={"text": "long prefix"})
        self.assertEqual(created.status_code, 201)
        cache_id = created.json()["cache_id"]

        listed = self.client.get("/v1/kv-caches")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["data"][0]["cache_id"], cache_id)

        completion = self.client.post(
            "/v1/completions",
            json={"kv_cache_id": cache_id, "prompt": "question", "max_tokens": 2},
        )
        self.assertEqual(completion.status_code, 200)
        self.assertEqual(completion.json()["choices"][0]["text"], "done")
        self.assertEqual(completion.json()["usage"]["cached_tokens"], 8)
        self.assertEqual(completion.json()["timings_ms"]["total"], 6.5)

        stats = self.client.get("/v1/kv-caches/stats")
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.json()["cache_count"], 0)

        deleted = self.client.delete(f"/v1/kv-caches/{cache_id}")
        self.assertTrue(deleted.json()["deleted"])

    def test_rejects_ambiguous_cache_input(self) -> None:
        response = self.client.post("/v1/kv-caches", json={"text": "a", "input_ids": [1, 2]})
        self.assertEqual(response.status_code, 422)

    def test_optional_api_key_protects_v1_routes(self) -> None:
        protected = TestClient(create_app(Settings(api_key="test-secret"), FakeBackend()))
        self.assertEqual(protected.get("/health").status_code, 200)
        self.assertEqual(protected.get("/v1/models").status_code, 401)
        authorized = protected.get("/v1/models", headers={"Authorization": "Bearer test-secret"})
        self.assertEqual(authorized.status_code, 200)


if __name__ == "__main__":
    unittest.main()
