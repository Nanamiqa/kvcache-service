# Backend 扩展指南

HTTP 层只认识 `KVCacheBackend`，因此本地引擎、私有化服务和云端 API 可以共用同一份业务
接口。扩展类应实现同步核心方法，或覆盖对应的异步方法；HTTP 层始终调用异步接口。

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

工厂在第一次 API 请求时惰性加载。`health()` 应返回不阻塞的本地快照；远端 readiness
检查放在 `ahealth()`。异步 backend 应复用连接池，并在 `aclose()` 中释放连接。

## 同步与异步接口

`KVCacheBackend` 为同步实现提供默认的 `asyncio.to_thread` 包装，因此 Transformers、文件
系统或简单 SDK 适配器可以只实现同步方法。高并发 HTTP backend 应直接覆盖：

- `ahealth`、`abuild_cache`、`aget_cache`、`alist_caches`；
- `adelete_cache`、`acache_stats`、`aprune_caches`；
- `acomplete`、`astream_complete` 和 `aclose`。

`astream_complete` 逐步产生 `CompletionChunk`。收到任务取消时必须立即关闭上游 response，
不要捕获并吞掉 `asyncio.CancelledError`。在发送第一个 token 之前可以安全地跨副本重试；一旦
已有数据发送给客户端，重试会造成重复文本，因此只能终止流并报告错误。

## 语义约定

- `BuildCacheCommand.text` 与 `input_ids` 二选一。
- `BuildCacheCommand.tenant_id`、`CompletionCommand.tenant_id` 必须参与所有 key、查询、列表、
  删除和远端 namespace；不能仅依赖调用方传入的 `cache_id` 做隔离。
- `request_id` 应透传至上游日志或 tracing header，但不得作为缓存 identity 的一部分。
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

## 生产实现检查表

- 对连接、首字节、流式读取和总请求分别设置有限超时。
- 限制连接池、在途请求和等待队列；不要让无界协程堆积在推理服务之前。
- 只对连接错误、超时和 5xx 计入断路器；调用方 4xx 不代表副本故障。
- 缓存路由同时考虑 affinity 和实时负载，避免热点 key 永久压在单一副本。
- 对相同 tenant/model/prefix 的并发预热使用单航班或分布式锁。
- 缓存 identity 包含不可变模型指纹；权重、tokenizer、量化或拓扑改变时切换 namespace。
- 暴露请求延迟、TTFT、cache hit tokens、upstream error、inflight 和逐副本健康指标。
- 将 KV 与逻辑前缀视为敏感数据，配置 TLS、密钥轮换、TTL、最小权限和审计。
