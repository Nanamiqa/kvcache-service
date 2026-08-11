# KV Cache Service

<div align="center">

**面向本地与私有化大语言模型推理的持久化 KV Cache 服务**

[![CI](https://github.com/Nanamiqa/kvcache-service/actions/workflows/ci.yml/badge.svg)](https://github.com/Nanamiqa/kvcache-service/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Version](https://img.shields.io/badge/version-0.3.0-blue)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

项目技术报告：[《KV Cache Service 技术报告：从推理中间态到企业知识资产》](<KV Cache Service 技术报告：从推理中间态到企业知识资产.pdf>)

KV Cache Service 是面向重复长前缀推理的缓存控制面与推理网关。0.3 版本提供两条运行路径：

- `transformers`：将每层 Key/Value 张量以 safetensors 持久化，适用于单机语义验证、研究和
  自定义 backend 开发；
- `vllm`：异步代理多个 vLLM 副本，以缓存亲和路由复用前缀，并可由 LMCache MP 扩展至
  CPU/NVMe/远端分层，适用于多 GPU 和高并发部署。

服务通过 FastAPI 提供缓存生命周期管理、OpenAI 风格 Completion API 和 SSE 流式输出，
内置租户隔离、准入控制、超时/取消、熔断、Prometheus 指标、健康探针及优雅退出。

## 目录

- [项目定位](#项目定位)
- [核心能力](#核心能力)
- [适用场景与方案选择](#适用场景与方案选择)
- [工作原理](#工作原理)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [API 概览](#api-概览)
- [持久化与一致性](#持久化与一致性)
- [容量评估](#容量评估)
- [配置](#配置)
- [兼容性约束](#兼容性约束)
- [安全建议](#安全建议)
- [扩展 Backend](#扩展-backend)
- [生产部署](#生产部署)
- [测试与性能评估](#测试与性能评估)
- [项目结构](#项目结构)
- [许可证](#许可证)

## 项目定位

自回归大语言模型在处理输入时，会为每个 token、每个注意力层生成 Key/Value 状态。对于
包含相同长前缀的多个请求，常规推理流程会重复计算该前缀，造成额外算力消耗并增加首 token
延迟（Time to First Token，TTFT）。

KV Cache Service 将这部分计算结果转换为可复用的缓存制品：

```text
首次请求：公共长前缀 ──prefill──> KV Cache ──持久化──> CPU / NVMe
后续请求：读取 KV Cache + 新增后缀 ──prefill──> decode ──> 生成结果
```

本项目缓存的是模型内部张量，而不是答案、文本向量或检索索引。因此，它不会替代 Redis、
向量数据库或 RAG 系统，也不能突破模型本身的上下文窗口。

## 核心能力

- **分块预填充**：按指定 token 数分块执行长前缀 prefill，避免一次性提交全部输入。
- **显式缓存复用**：通过稳定的 `cache_id` 创建、查询、加载和删除 KV Cache。
- **内容寻址**：`cache_id` 由模型指纹与精确 token 序列共同确定；相同输入不会重复构建。
- **安全持久化**：使用 `safetensors` 保存张量，避免反序列化任意 Python 对象。
- **一致性保护**：缓存发布采用临时目录和原子替换，并默认校验张量文件 SHA-256。
- **精确前缀验证**：`prompt_mode=full` 时逐 token 验证完整输入是否以缓存前缀开头。
- **容量治理**：支持 TTL、存储容量上限、LRU 淘汰和主动清理。
- **跨进程协调**：进程内锁与文件锁共同保护缓存的读、写、删操作。
- **访问控制**：可为全部 `/v1/*` 路由启用 Bearer Token 或 `X-API-Key` 鉴权。
- **阶段耗时统计**：返回缓存加载、输入处理、prefill、decode 和端到端耗时。
- **可插拔 Backend**：可接入本地推理引擎、私有服务或云厂商逻辑缓存接口。
- **异步流式推理**：完整代理 SSE，客户端断开时取消上游请求，并限制请求截止时间。
- **多副本缓存亲和路由**：结合 rendezvous affinity、在途负载与 EWMA 延迟选择 vLLM。
- **过载保护**：全局/租户并发上限、排队超时和每租户 token bucket 限流。
- **故障隔离**：vLLM 重试、主动探测和逐副本熔断，避免故障副本拖垮入口。
- **水平扩展控制面**：Redis 共享逻辑索引、TTL 和分布式预热锁；SQLite 供单网关使用。
- **多租户安全**：租户 API key、租户参与 cache identity、管理员状态接口。
- **可观察性**：Prometheus 请求、缓存、token、后端延迟/健康度指标及标准探针。
- **工程化支持**：提供轻量网关镜像、生产 Compose、Kubernetes、测试与压测脚本。

## 适用场景与方案选择

典型适用场景包括：

- 对同一份长文档、代码上下文、合同或知识材料反复提问；
- 多个请求共享固定的系统提示词、工具定义或 few-shot 示例；
- 私有化 Agent 工作流中存在复用率较高的稳定前缀；
- 研究磁盘加载 KV Cache 与重新 prefill 之间的性能边界；
- 验证显式缓存 ID、缓存生命周期及模型兼容性语义；
- 为自研推理引擎或云端 Prompt Cache 构建统一 HTTP 控制面。

建议根据实际负载选择实现路径：

| 场景 | 推荐方案 | 说明 |
|---|---|---|
| 功能验证、单机低并发、原始 KV 落盘 | Transformers backend | `KVCACHE_BACKEND=transformers` |
| 私有化生产、多 GPU、高并发、分层存储 | vLLM backend + LMCache MP | 参见 [`deploy/`](deploy/README.md) |
| 已有私有推理平台 | 自定义 `KVCacheBackend` | 复用本项目 HTTP 契约 |
| 公有云模型 API | 逻辑缓存 backend | 映射厂商 Prompt Cache 语义 |
| 短输入或低前缀复用率 | 普通推理 | 缓存加载成本可能高于重算 |

## 工作原理

### 系统架构

```mermaid
flowchart LR
    C["Client / SDK"] --> A["Async FastAPI gateway"]
    A --> L["Tenant auth + admission"]
    L --> B["KVCacheBackend"]
    B --> T["Transformers reference"]
    T --> S["Atomic safetensors store"]
    B --> R["Cache-affine replica router"]
    R --> V0["vLLM replica 0"]
    R --> V1["vLLM replica 1..N"]
    A --> D["Redis logical index"]
    V0 --> M["LMCache MP tiers"]
    V1 --> M
```

HTTP 层仅依赖 `KVCacheBackend` 抽象，不直接依赖 Transformers。替换 backend 时，缓存管理
和 Completion API 无需改动。

### 缓存构建流程

1. 接收公共前缀文本或调用方提供的 `input_ids`。
2. 使用目标模型的 tokenizer 得到精确 token 序列。
3. 校验 token 数未超过模型上下文窗口。
4. 按 `chunk_size` 分块执行 prefill，并持续累积 `past_key_values`。
5. 计算模型指纹、前缀哈希和内容寻址 `cache_id`。
6. 将每层 Key/Value、前缀 token IDs 及下一 token logits 写入 `safetensors`。
7. 生成元数据和校验和，以原子方式发布缓存目录。

### 缓存复用流程

1. 根据 `cache_id` 读取元数据与张量文件。
2. 校验缓存未过期、文件未损坏且模型指纹一致。
3. 将缓存张量恢复到推理设备。
4. 可选地验证完整 prompt 与已缓存 token 前缀完全一致。
5. 仅对新增后缀执行 prefill，随后进入正常 token 解码流程。

## 系统要求

| 项目 | 要求 |
|---|---|
| Python | 3.9 及以上；推荐 3.11 或 3.12 |
| 操作系统 | Linux、macOS 或 Windows |
| 推理框架 | PyTorch 2.2–2.x、Transformers 4.56–4.57 |
| 模型类型 | Decoder-only Causal Language Model |
| 设备 | CPU、CUDA 或 MPS；由 `KVCACHE_DEVICE` 控制 |
| 存储 | 本地文件系统；建议使用容量充足的高速 NVMe |

默认示例模型为 `Qwen/Qwen2.5-0.5B-Instruct`。首次实际构建缓存时，Transformers 会从
Hugging Face 下载模型。离线环境可将 `KVCACHE_MODEL` 指向本地模型目录，并设置
`KVCACHE_LOCAL_FILES_ONLY=true`。

## 快速开始

### 方式一：本地安装

克隆仓库并创建虚拟环境：

```bash
git clone https://github.com/Nanamiqa/kvcache-service.git
cd kvcache-service
python -m venv .venv
```

Linux 或 macOS：

```bash
source .venv/bin/activate
pip install -e '.[local]'
cp .env.example .env
kvcache-server
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[local]"
Copy-Item .env.example .env
kvcache-server
```

服务默认监听 `http://localhost:8080`。默认配置可直接启动；`.env.example` 用于展示配置项，
如需覆盖默认值，请在启动进程前把相应变量导入当前环境。

### 方式二：Docker Compose

```bash
docker compose up --build
```

该 Compose 配置用于 CPU 功能演示。GPU 部署应使用与宿主 CUDA 版本匹配的 PyTorch、vLLM
或厂商基础镜像，不应直接把演示镜像用于生产环境。

### 方式三：多 GPU 生产拓扑

```bash
cp .env.production.example .env.production
# 填写模型、固定镜像版本、Redis 密码和租户密钥
docker compose --env-file .env.production -f compose.production.yaml up -d --build
```

该拓扑启动无 Torch 依赖的网关、Redis、LMCache MP 和两个 vLLM GPU 副本。完整的版本固定、
容量规划与 Kubernetes 部署流程见 [`deploy/README.md`](deploy/README.md)。

### 验证服务

```bash
curl http://localhost:8080/health
```

启动后还可访问以下地址：

- OpenAPI 文档：`http://localhost:8080/docs`
- OpenAPI Schema：`http://localhost:8080/openapi.json`
- 存活探针：`http://localhost:8080/livez`
- 就绪探针：`http://localhost:8080/readyz`
- Prometheus：`http://localhost:8080/metrics`

## 使用方式

### 1. 创建公共前缀缓存

```bash
curl -X POST http://localhost:8080/v1/kv-caches \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "这里放会被多个请求复用的长文档或系统前缀。\n\n",
    "chunk_size": 512
  }'
```

响应中的 `cache_id` 由模型指纹和精确 token 序列共同决定。同一模型环境下重复提交相同
输入会返回已有缓存，不会再次执行 prefill。

### 2. 从缓存继续生成

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "kv_cache_id": "<cache_id>",
    "prompt": "请概括上面的内容。",
    "max_tokens": 128,
    "temperature": 0
  }'
```

响应中的关键字段：

- `usage.cached_tokens`：本次跳过 prefill 的前缀 token 数；
- `usage.prompt_tokens`：缓存前缀与新增后缀的总 token 数；
- `timings_ms.cache_load`：缓存读取与恢复耗时；
- `timings_ms.prefill`：新增后缀 prefill 耗时；
- `timings_ms.decode`：逐 token 解码耗时；
- `timings_ms.total`：端到端总耗时。

不传 `kv_cache_id` 时，`/v1/completions` 会执行普通 Completion。

启用流式输出时设置 `"stream": true`，服务返回 OpenAI 风格 SSE，并在终止事件中附带
usage：

```bash
curl -N http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"介绍一下 KV Cache","max_tokens":128,"stream":true}'
```

### 3. 使用完整 Prompt

默认 `prompt_mode=suffix`，即 `prompt` 仅包含缓存前缀之后的新增内容。如果调用方持有完整
prompt，可指定 `prompt_mode=full`：

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "kv_cache_id": "<cache_id>",
    "prompt": "<原始完整前缀>请概括上面的内容。",
    "prompt_mode": "full",
    "max_tokens": 128
  }'
```

服务会重新分词并逐 token 校验缓存前缀，只计算剩余后缀。前缀不一致时返回
`409 cache_incompatible`，不会在不兼容的 KV Cache 上继续推理。

> [!NOTE]
> 对 token 边界有严格要求时，建议在构建和调用阶段都传入 `input_ids`。分别编码
> `prefix` 与 `suffix` 的结果不一定等同于一次性编码 `prefix + suffix`，尤其是在前缀末尾
> 没有空格、换行或模板边界时。

### 4. 启用租户鉴权

单租户兼容模式可设置：

```bash
export KVCACHE_API_KEY='replace-with-a-strong-secret'
```

之后访问 `/v1/*` 时需携带以下任一请求头：

```http
Authorization: Bearer replace-with-a-strong-secret
```

或：

```http
X-API-Key: replace-with-a-strong-secret
```

多租户模式使用 tenant 到密钥的 JSON 映射，并在每次请求中同时发送租户头：

```bash
export KVCACHE_API_KEYS='{"tenant-a":"secret-a","tenant-b":"secret-b"}'
curl -H 'X-Tenant-ID: tenant-a' -H 'Authorization: Bearer secret-a' \
  http://localhost:8080/v1/kv-caches
```

同一前缀在不同租户下生成不同 `cache_id`，所有查询、删除和统计也按租户隔离。`/livez`、
`/readyz`、`/health` 和 `/metrics` 保持公开以供编排与监控；应通过网络策略限制这些端点。
管理员状态接口需要单独的 `KVCACHE_ADMIN_API_KEY`。

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/livez` | 进程存活探针，不访问后端 |
| `GET` | `/readyz` | 检查 draining 状态和可用推理副本 |
| `GET` | `/health` | 返回 backend、模型和逐副本状态 |
| `GET` | `/metrics` | Prometheus 指标 |
| `GET` | `/v1/models` | 返回 OpenAI 风格模型列表 |
| `GET` | `/v1/admin/status` | 管理员可见的后端、缓存和限流状态 |
| `POST` | `/v1/kv-caches` | 创建/预热公共前缀 |
| `GET` | `/v1/kv-caches` | 列出有效缓存 |
| `GET` | `/v1/kv-caches/stats` | 返回缓存数量、逻辑大小、磁盘占用及治理配置 |
| `POST` | `/v1/kv-caches/prune` | 清理过期或超出容量限制的缓存 |
| `GET` | `/v1/kv-caches/{id}` | 查询指定缓存元数据 |
| `DELETE` | `/v1/kv-caches/{id}` | 删除指定缓存 |
| `POST` | `/v1/completions` | 普通/缓存续推；支持 `stream=true` SSE |

API 使用 Pydantic 严格校验请求字段。服务错误采用稳定结构返回：

```json
{
  "error": {
    "code": "cache_incompatible",
    "message": "Full prompt does not start with the exact cached token prefix",
    "type": "invalid_request_error"
  }
}
```

## 持久化与一致性

默认情况下，每个缓存对应一个独立目录：

```text
data/kv-cache/<cache_id>/
├── metadata.json
└── tensors.safetensors
```

`metadata.json` 记录以下信息：

- 缓存 Schema 版本；
- backend、模型 ID 与模型指纹；
- 前缀 token 数量及 SHA-256；
- K/V 层数、dtype 与逻辑字节数；
- 创建时间、过期时间和构建分块大小；
- `tensors.safetensors` 文件校验和。

`tensors.safetensors` 包含每层 K/V 张量、精确前缀 token IDs 以及前缀结束位置的下一 token
logits。因此，即使后缀为空，服务仍可从缓存状态继续生成。

存储层提供以下一致性保证：

- 在临时目录完整写入后，以原子方式发布缓存；
- 默认在首次加载或文件发生变化后校验 SHA-256；
- 同一进程内缓存已验证文件的大小与修改时间，避免重复扫描大型张量文件；
- 使用进程内可重入锁和跨进程文件锁保护读、写、删；
- 实际加载缓存时刷新目录访问时间，供 LRU 淘汰使用；
- 读取到过期缓存时自动删除；也可通过 prune API 主动清理；
- 新缓存大于总配额时保留该缓存并在统计中反映超额，避免返回立即失效的缓存 ID。

## 容量评估

KV Cache 的理论大小可按以下公式估算：

```text
2 × 层数 × KV heads × head dimension × token 数 × 每元素字节数 × batch size
```

其中系数 `2` 分别代表 Key 与 Value。以 28 层、4 个 KV heads、head dimension 为 128、
FP16、batch size 为 1 的 GQA 模型为例，理论占用约为 56 KiB/token；128K tokens 约需
7 GiB，尚未包含元数据、文件系统和运行时搬运开销。

实际部署前应同时评估：

- 公共前缀长度、数量与平均复用次数；
- KV dtype、模型层数及注意力结构；
- 磁盘容量、顺序读带宽与并发读取能力；
- CPU 内存以及 CPU→GPU 的有效带宽；
- TTL、LRU 淘汰和租户隔离带来的冗余；
- 缓存加载延迟是否低于重新 prefill 的计算延迟。

## 配置

完整示例参见 [`.env.example`](.env.example)。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `KVCACHE_BACKEND` | `transformers` | 内置 backend 或 `python.module:factory` |
| `KVCACHE_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face 模型 ID 或本地目录 |
| `KVCACHE_MODEL_REVISION` | `main` | 模型 revision |
| `KVCACHE_DEVICE` | `auto` | `auto`、`cpu`、`cuda`、`mps` 等 |
| `KVCACHE_DTYPE` | `auto` | `float16`、`bfloat16`、`float32` 等 |
| `KVCACHE_MODEL_FINGERPRINT` | 空 | 部署方提供的不可变权重标识 |
| `KVCACHE_TRUST_REMOTE_CODE` | `false` | 是否允许执行模型仓库 remote code |
| `KVCACHE_LOCAL_FILES_ONLY` | `false` | 是否仅从本地读取模型文件 |
| `KVCACHE_STORE_DIR` | `./data/kv-cache` | 缓存根目录 |
| `KVCACHE_VERIFY_CHECKSUM` | `true` | 加载缓存时是否校验张量文件 |
| `KVCACHE_CACHE_TTL_SECONDS` | `0` | 缓存有效期；`0` 表示不自动过期 |
| `KVCACHE_MAX_STORE_BYTES` | `0` | 缓存总容量上限；`0` 表示不限制 |
| `KVCACHE_MAX_CONTEXT_TOKENS` | `0` | 显式上下文上限；`0` 表示从模型配置读取 |
| `KVCACHE_DEFAULT_CHUNK_SIZE` | `512` | 默认 prefill 分块 token 数 |
| `KVCACHE_MAX_NEW_TOKENS` | `2048` | 单次请求允许生成的最大 token 数 |
| `KVCACHE_API_KEY` | 空 | `/v1/*` 路由的可选鉴权密钥 |
| `KVCACHE_API_KEYS` | 空 | tenant 到 API key 的 JSON 映射；启用多租户鉴权 |
| `KVCACHE_ADMIN_API_KEY` | 空 | `/v1/admin/*` 的独立密钥 |
| `KVCACHE_TENANT_HEADER` | `X-Tenant-ID` | 租户标识请求头 |
| `KVCACHE_REQUEST_TIMEOUT_SECONDS` | `600` | 单请求与流式会话截止时间 |
| `KVCACHE_ADMISSION_QUEUE_TIMEOUT_SECONDS` | `5` | 准入排队超时 |
| `KVCACHE_MAX_CONCURRENT_REQUESTS` | `256` | 网关全局在途上限 |
| `KVCACHE_MAX_CONCURRENT_PER_TENANT` | `32` | 单租户在途上限 |
| `KVCACHE_RATE_LIMIT_TOKENS_PER_MINUTE` | `0` | 单租户 token 预算；`0` 为关闭 |
| `KVCACHE_REDIS_URL` | 空 | 多网关共享逻辑索引；空值使用 SQLite |
| `KVCACHE_VLLM_ENDPOINTS` | `http://127.0.0.1:8000` | 逗号分隔的 vLLM endpoint |
| `KVCACHE_VLLM_API_KEY` | 空 | 网关访问 vLLM 的 Bearer key |
| `KVCACHE_VLLM_MAX_RETRIES` | `1` | 非流式请求跨副本重试次数 |
| `KVCACHE_VLLM_CIRCUIT_BREAKER_FAILURES` | `3` | 打开逐副本断路器的连续失败数 |
| `KVCACHE_METRICS_ENABLED` | `true` | 是否挂载 `/metrics` |
| `KVCACHE_HOST` | `0.0.0.0` | HTTP 监听地址 |
| `KVCACHE_PORT` | `8080` | HTTP 监听端口 |
| `KVCACHE_LOG_LEVEL` | `info` | Uvicorn 日志级别 |

## 兼容性约束

原始 KV Cache 只有在以下条件保持兼容时才能安全复用：

- 模型权重及其不可变 revision；
- tokenizer、词表和 chat template；
- dtype、量化方式和计算布局；
- 注意力类型、KV heads、head dimension 与 RoPE 参数；
- tensor-parallel 拓扑和 cache layout；
- 精确 token 序列及其位置。

Hub 模型优先使用已解析 commit 标识模型来源；本地模型目录会根据配置、tokenizer、remote
code 内容以及权重文件清单生成指纹。若本地权重可能被原地覆盖，应设置
`KVCACHE_MODEL_FINGERPRINT` 为不可变发布版本或完整权重哈希，防止误用旧缓存。

当前 Transformers backend 有意限制为：

- batch size 为 1；
- decoder-only Causal LM；
- 完整的 Dynamic 或 legacy K/V layout；
- 单个模型由进程内锁串行访问；
- 不支持滑动窗口、hybrid attention、Mamba/线性 attention 和 tensor parallel。

分块 prefill 不能突破模型的位置编码或上下文窗口。输入超过模型上限时，应使用长上下文
模型，或在业务层采用 RAG、文本切分、摘要等策略。

## 安全建议

KV 张量和 token IDs 应按照原始提示词同等级别进行保护。虽然 KV Cache 不是明文文本，仍不
应将其视为匿名或无敏感信息的数据。

生产环境至少应落实以下措施：

- 将缓存目录放置在受控的私有文件系统中，不使用公开对象存储；
- 对缓存盘启用静态加密，并保护备份与快照；
- 为不同租户使用独立 namespace、目录或存储实例；
- 使用网关、网络策略和细粒度身份认证保护服务；
- 配置合理的 TTL、容量上限、淘汰策略和安全删除机制；
- 默认保持 `KVCACHE_TRUST_REMOTE_CODE=false`；
- 定期轮换 API Key，并避免在日志或仓库中提交密钥；
- 监控异常缓存加载、校验失败和跨租户访问行为。

内置 API Key 适合开发和受控网络中的基础保护，不等同于完整的多租户认证与授权系统。

## 扩展 Backend

实现 [`KVCacheBackend`](src/kvcache_service/backend.py) 即可接入新的本地引擎、私有推理
服务或公有云逻辑缓存。

Backend 工厂应接收 `Settings` 并返回 `KVCacheBackend` 实例：

```python
def create_backend(settings: Settings) -> KVCacheBackend:
    return MyBackend(settings)
```

通过环境变量加载：

```bash
export KVCACHE_BACKEND=my_package.provider:create_backend
kvcache-server
```

仓库提供了可运行骨架和详细约定：

- [`examples/custom_backend.py`](examples/custom_backend.py)
- [`docs/backend-extension.md`](docs/backend-extension.md)

对于能够访问模型内部张量的引擎，backend 可以直接保存和恢复原始 KV。对于不开放原始 KV
的公有云 API，可将 `cache_id` 映射为厂商缓存名称、Prompt Cache Key 或公共前缀记录。

## 生产部署

0.3 内置 `vllm` backend，生产环境不再由应用层显式传输大型 K/V 文件。其职责边界为：

- 使用 vLLM 管理 GPU 内 KV block 和 Automatic Prefix Caching；
- 使用 LMCache MP 将缓存扩展至 CPU、NVMe 或远端存储；
- 网关保存租户隔离的逻辑前缀，并以缓存亲和、负载和延迟选择副本；
- Redis 在多个网关之间共享逻辑索引、TTL 和防重复预热锁；
- 通过准入控制、重试、熔断、探针和优雅退出保护数据平面。

参考配置与启动命令见 [`deploy/README.md`](deploy/README.md)。上线前应固定 vLLM、CUDA、
PyTorch 与 LMCache 的兼容版本，不应在生产环境使用浮动的 `latest` 依赖。

建议重点监控：

- TTFT 与端到端延迟；
- cache hit tokens 与命中率；
- cache load/store latency；
- GPU、CPU 与磁盘容量；
- eviction、过期和校验失败次数；
- CPU→GPU 与磁盘实际读取带宽。

只有当“重新计算前缀的成本”高于“读取缓存并搬运至推理设备的成本”时，持久化 KV Cache
才会产生实际收益。应使用真实模型、真实存储和真实请求分布进行基准测试。

## 测试与性能评估

安装开发依赖并执行完整检查：

```bash
pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest
python -m build
```

测试默认使用伪 Causal LM，不下载模型，覆盖以下行为：

- 分块 prefill、张量落盘、重新加载和缓存续推；
- Transformers `past_key_values` 兼容性；
- 完整 prompt 的精确 token 前缀验证；
- 无后缀情况下从已保存 logits 继续生成；
- SHA-256 校验和损坏检测；
- TTL、LRU 配额和主动清理；
- 多租户鉴权、隔离及稳定错误结构；
- 异步 SSE、vLLM 协议映射、缓存亲和与健康检查；
- Redis 共享索引、分布式预热锁、并发和 token 准入控制。

GitHub Actions 会在推送和 Pull Request 时执行 Ruff、测试及 wheel 构建。

服务启动后，可使用基准脚本比较冷 prefill 与缓存复用耗时：

```bash
python scripts/benchmark.py ./long-document.txt --runs 3 --max-tokens 64
```

建议重点比较输出中的 `cache_load`、`prefill` 和 `total`，并使用多种前缀长度与复用次数进行
评估。

生产链路可使用并发脚本记录 TTFT、P95/P99、请求吞吐和输出 token 吞吐：

```bash
python scripts/load_test.py --requests 500 --concurrency 64 --max-tokens 128
```

## 项目结构

```text
kvcache-service/
├── src/kvcache_service/
│   ├── app.py                  # FastAPI 路由、鉴权与错误处理
│   ├── backend.py              # Backend 抽象与动态加载
│   ├── transformers_backend.py # Transformers 参考实现
│   ├── vllm_backend.py         # 异步多副本 vLLM 数据平面
│   ├── router.py               # 缓存亲和、负载反馈与熔断
│   ├── admission.py            # 并发与 token 准入控制
│   ├── logical_store.py        # SQLite 逻辑缓存索引
│   ├── redis_store.py          # Redis 共享索引与分布式锁
│   ├── metrics.py              # Prometheus 指标
│   ├── store.py                # safetensors 持久化与容量治理
│   ├── cache_codec.py          # K/V 张量编码与恢复
│   ├── api_models.py           # API 请求与响应模型
│   └── config.py               # 环境变量配置
├── tests/                      # 单元及集成测试
├── examples/                   # 自定义 Backend 示例
├── docs/                       # 扩展文档
├── deploy/                     # vLLM + LMCache 生产化参考
├── scripts/                    # 冷/热路径基准及并发负载测试
├── deploy/kubernetes/          # 网关、vLLM、LMCache Operator 示例
├── compose.yaml                # CPU 演示环境
├── compose.production.yaml     # 双 GPU 生产参考拓扑
├── Dockerfile.gateway          # 无 Torch 网关镜像
├── Dockerfile                  # Transformers 演示镜像
└── pyproject.toml              # Python 包及工具配置
```

## 相关资料

- [Transformers cache strategies](https://huggingface.co/docs/transformers/kv_cache)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [LMCache architecture](https://docs.lmcache.ai/developer_guide/architecture.html)
- [LMCache MP deployment](https://docs.lmcache.ai/mp/deployment.html)

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
