# vLLM + LMCache 生产部署参考

参考实现适合验证显式 KV 序列化语义；生产私有化服务更适合让 vLLM 管理 GPU block，由
LMCache 做 CPU/NVMe/远端分层。客户端仍调用 OpenAI-compatible API，并通过重复的 token
前缀自动命中缓存，不需要在网络上传输 KV tensor。

下面命令基于 LMCache 官方 V1 connector。vLLM、CUDA 和 LMCache 的兼容版本变化较快，
部署前请按官方 compatibility matrix 固定镜像和依赖版本，不要在生产中使用浮动 latest。

## 方式一：单进程快速验证

```bash
python -m venv .venv-lmcache
source .venv-lmcache/bin/activate
pip install lmcache vllm

export LMCACHE_CONFIG_FILE="$PWD/deploy/lmcache.yaml"
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 32768 \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

连续两次发送具有相同长前缀的请求，第二次会从 LMCache 恢复已对齐的 token chunks：

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"<相同长前缀>问题一","max_tokens":64}'

curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","prompt":"<相同长前缀>问题二","max_tokens":64}'
```

## 方式二：独立 LMCache server

需要多 vLLM engine 共享缓存时，使用 MP connector：

```bash
lmcache server \
  --host 0.0.0.0 \
  --port 5555 \
  --l1-size-gb 100 \
  --eviction-policy LRU
```

另一个进程启动 vLLM：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555}}'
```

## 上线前检查

- 模型 revision、tokenizer/chat template、量化和 TP 拓扑固定，不兼容变更使用新 namespace。
- cache key 必须包含 tenant，严禁不同租户通过相同 prompt hash 共享 KV。
- NVMe 读带宽和 CPU→GPU 带宽必须实测；低复用率时，落盘成本可能高于重新 prefill。
- 接入 `/metrics`，重点观察 TTFT、cache hit tokens、load/store latency、GPU/CPU/disk 容量和
  eviction。
- 对缓存目录做加密、访问控制、TTL 和安全删除；KV 应视作敏感提示词数据。
- 优先按 token chunk 做自动前缀匹配，而不是为每个完整文档复制一份 cache。

官方文档：

- [LMCache quickstart](https://docs.lmcache.ai/getting_started/quickstart.html)
- [LMCache configuration](https://docs.lmcache.ai/api_reference/configurations.html)
- [LMCache architecture](https://docs.lmcache.ai/developer_guide/architecture.html)
