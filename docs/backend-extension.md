# Backend 扩展指南

HTTP 层只认识 `KVCacheBackend`，因此本地引擎、私有化服务和云端 API 可以共用同一份业务
接口。扩展类需要实现缓存生命周期、completion 和 health 六个方法。

## 两类适配方式

### 原始 KV backend

适用于能拿到模型内部张量的推理引擎。`build_cache` 执行 prefill 并保存 K/V，`complete`
按 `cache_id` 注入缓存。兼容性校验至少应包含：

- 模型权重不可变 revision 或权重 hash；
- tokenizer 与 chat template 版本；
- dtype/量化参数；
- attention layout、RoPE 参数和 tensor-parallel 拓扑；
- token 序列及其位置。

### 逻辑缓存 backend

公有云一般不返回原始 KV。适配器可以把本服务的 `cache_id` 映射为厂商的 cache name、
prompt-cache key 或公共前缀记录，并在 `complete` 中调用远端 API。这种 backend 的
`tensor_bytes` 和 `layer_count` 可以为 0，但必须在 metadata 或自己的数据库里保留租户、
模型版本和过期时间。

## 加载约定

把 backend 工厂写成普通 Python callable：

```python
def create_backend(settings: Settings) -> KVCacheBackend:
    return MyBackend(settings)
```

再设置：

```bash
KVCACHE_BACKEND=my_package.provider:create_backend kvcache-server
```

工厂在第一次 API 请求时惰性加载。建议 `health()` 不做昂贵的模型加载；真正的远端
readiness 检查可在 backend 内按需缓存。

## 语义约定

- `BuildCacheCommand.text` 与 `input_ids` 二选一。
- `CompletionCommand.prompt` 是**已缓存前缀之后的后缀**，不是完整 prompt。
- 当 `CompletionCommand.prompt_mode == "full"` 时，backend 应验证并剥离完整输入中的缓存
  前缀；无法支持时应明确返回 `CacheCompatibilityError`。
- 没有 `cache_id` 时，`complete` 应执行普通生成。
- backend 应保证同一个 `cache_id` 对应不可变内容。
- 不兼容返回 `CacheCompatibilityError`，不存在返回 `CacheNotFoundError`，上下文越界返回
  `ContextLengthError`；HTTP 层会生成稳定错误结构。
- 如果云 API 只能自动缓存而不能显式创建，`build_cache` 可以保存逻辑前缀，首次
  `complete` 负责预热，后续请求返回实际命中统计。
- `CompletionResult.timings_ms` 至少应包含 `cache_load`、`input_processing`、`prefill`、
  `decode` 和 `total`，无法拆分的远端 API 可以把未知阶段设为 `0`。
