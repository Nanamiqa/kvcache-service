# 生产部署指南

本目录提供 KV Cache Service 0.3 的生产参考拓扑。生产模式下，网关不直接持有 GPU
张量：vLLM 负责连续批处理、GPU KV block 和 Automatic Prefix Caching，LMCache MP
负责 CPU/NVMe/远端分层，Redis 保存租户隔离的逻辑缓存索引。网关可以独立水平扩容。

```text
Client
  │  OpenAI-compatible API / SSE
  ▼
KV Cache Gateway (N replicas)
  ├── Redis: logical prefix index, TTL, build lock
  └── cache-affine routing + load feedback + circuit breaker
          │
          ├── vLLM replica 0 ── LMCache MP (node local)
          └── vLLM replica 1 ── LMCache MP (node local)
```

## 选择部署方式

| 场景 | 入口 | 用途 |
|---|---|---|
| 单机双 GPU 验证 | `compose.production.yaml` | 验证完整生产链路与故障切换 |
| Kubernetes | `kubernetes/` | 多副本网关、vLLM StatefulSet、LMCache Operator |
| 单机张量语义验证 | 根目录 `compose.yaml` | Transformers 参考实现，不用于高并发 |

所有生产镜像、模型 revision 和 Python/CUDA 依赖必须固定到版本或 digest。示例中的
`REPLACE_ME`、`PINNED_VERSION` 和密钥占位符必须在应用前替换。

## Docker Compose

复制环境模板并填写不可变版本及密钥：

```bash
cp .env.production.example .env.production
docker compose --env-file .env.production -f compose.production.yaml config
docker compose --env-file .env.production -f compose.production.yaml up -d --build
```

该拓扑默认使用 GPU 0 和 1。LMCache 的 `--max-gpu-workers` 至少应等于共享该 server
的 vLLM 实例数；L1 容量需为操作系统、模型进程和突发负载保留余量。

验证：

```bash
curl http://localhost:8080/livez
curl http://localhost:8080/readyz
curl http://localhost:8080/metrics
curl -H "X-Admin-Key: $KVCACHE_ADMIN_API_KEY" \
  http://localhost:8080/v1/admin/status
```

## Kubernetes

推荐使用 LMCache Operator 管理每个 GPU 节点上的 MP server。操作顺序：

1. 安装 GPU Operator、LMCache Operator、Redis（建议托管或高可用实例）和可选的
   Prometheus Operator。
2. 创建独立 namespace；LMCache CUDA IPC 需要 `hostIPC`，应在受信任节点池中按组织
   安全策略放行。
3. 从 `secret.example.yaml` 创建真实 Secret，切勿提交实际密钥。
4. 应用 `gateway.yaml` 中的 ConfigMap，随后先应用
   `data-plane.operator.example.yaml` 的 `LMCacheEngine` 部分。
5. 等待 Operator 生成 `lmcache-connection` ConfigMap，再应用同文件中的 vLLM
   StatefulSet、Service 和 PDB。
6. 应用 `gateway.yaml` 的其余资源；如使用 Prometheus Operator，再应用
   `monitoring.example.yaml`。

vLLM StatefulSet 使用稳定 DNS 名 `vllm-0.vllm-headless` 和
`vllm-1.vllm-headless`，与网关 ConfigMap 中的 endpoint 对应。扩容数据平面时需同步更新
`KVCACHE_VLLM_ENDPOINTS`。若不希望手工维护 endpoint 列表，可在外部控制器中根据 Pod
就绪状态生成 ConfigMap，并滚动重启网关。

LMCache Operator 会为同节点 vLLM 提供 node-local Service 和连接 ConfigMap；不要在
vLLM Pod 中额外挂载私有 `/dev/shm`，否则会遮蔽 `hostIPC` 所需的宿主共享内存。

## 容量与调度基线

- vLLM：先通过真实负载调节 `--max-num-batched-tokens`、`--max-num-seqs`、
  `--gpu-memory-utilization`、chunked prefill 和 TP/DP。不要仅以吞吐量峰值定参。
- LMCache：L1 容量按热点工作集估算；仅当 NVMe/远端读取加搬运快于重算 prefill 时启用
  L2。`--max-gpu-workers` 不得少于共享实例数。
- 网关：CPU/内存开销较小，可根据队列等待、事件循环延迟和连接数扩容；并发上限应略低于
  后端可持续容量，令过载在入口快速失败。
- Redis：启用认证、TLS（跨主机时）、持久化/复制和内存告警。Redis 只存逻辑前缀与元数据，
  不存 GPU KV tensor。

## 发布与回滚

模型权重、revision、tokenizer、chat template、量化方式、KV dtype 或 TP 布局发生不兼容
变化时，必须使用新的 `KVCACHE_MODEL_FINGERPRINT` 和 Redis key prefix。先部署新数据平面，
待 `/readyz` 通过后再滚动网关；旧缓存让 TTL 自然回收。不要让两个不兼容版本共享逻辑
namespace。

滚动终止时，编排器先将 Pod 从 Service 摘除，`preStop` 留出传播时间，网关进入 draining
后拒绝新请求并等待在途请求，最后关闭 HTTP/Redis 连接。长流式请求的最大持续时间由
`KVCACHE_REQUEST_TIMEOUT_SECONDS` 限制。

## 可观察性与告警

网关在 `/metrics` 暴露请求量、延迟、拒绝、在途请求、各 vLLM endpoint 健康度/延迟、
缓存操作和 token 指标。LMCache MP 的 `lmcache server` 在 HTTP 端口（默认 8080）的
`/metrics` 暴露指标；vLLM 在自身 `/metrics` 暴露调度与 GPU KV 指标。

建议至少为以下事件告警：

- `/readyz` 失败或健康 vLLM 副本数为 0；
- 429/503/504 比率、排队拒绝率或断路器打开持续升高；
- TTFT、P95/P99 总延迟和流式中断率恶化；
- cache hit tokens 比例骤降或 LMCache load/store 延迟升高；
- GPU KV 使用率、LMCache L1 水位、Redis 内存和 eviction 接近上限；
- 单租户 token 预算、并发或错误率异常。

## 安全边界

KV Cache 应按原始提示词敏感级别保护。生产环境必须启用租户 API key 或接入上游身份
代理、网络策略、TLS、密钥轮换、缓存 TTL、静态加密和审计。网关的 `cache_id` 已包含租户
和模型指纹，Redis key 也按租户散列隔离；这不替代组织级授权与密钥管理。

## 兼容性资料

- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [LMCache MP deployment](https://docs.lmcache.ai/mp/deployment.html)
- [LMCache Kubernetes Operator](https://docs.lmcache.ai/mp/operator.html)
- [LMCache observability](https://docs.lmcache.ai/mp/observability/index.html)
