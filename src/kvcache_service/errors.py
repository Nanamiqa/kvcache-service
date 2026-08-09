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


class ContextLengthError(KVCacheError):
    status_code = 422
    code = "context_length_exceeded"


class BackendConfigurationError(KVCacheError):
    status_code = 503
    code = "backend_configuration_error"
