"""Domain exceptions converted to stable API errors."""


class KVCacheError(Exception):
    """Base class for expected service errors."""

    status_code = 400
    code = "kv_cache_error"


class CacheNotFoundError(KVCacheError):
    status_code = 404
    code = "cache_not_found"


class CacheCompatibilityError(KVCacheError):
    status_code = 409
    code = "cache_incompatible"


class CacheExpiredError(KVCacheError):
    status_code = 410
    code = "cache_expired"


class ContextLengthError(KVCacheError):
    status_code = 422
    code = "context_length_exceeded"


class BackendConfigurationError(KVCacheError):
    status_code = 503
    code = "backend_configuration_error"


class BackendUnavailableError(KVCacheError):
    status_code = 503
    code = "backend_unavailable"


class RequestOverloadedError(KVCacheError):
    status_code = 429
    code = "rate_limit_exceeded"


class RequestTimeoutError(KVCacheError):
    status_code = 504
    code = "request_timeout"


class UpstreamAPIError(KVCacheError):
    code = "upstream_request_error"

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
