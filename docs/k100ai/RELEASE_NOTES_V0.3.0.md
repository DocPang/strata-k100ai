# Strata for K100AI v0.3.0 发布说明

发布对象：`DocPang/strata-k100ai`  
标签：`v0.3.0`  
适用平台：Hygon K100AI / gfx928，Linux x86_64  
发布状态：生产可用

v0.3.0 面向 Hygon K100AI 提供 DeepSeek-V4.1-Flash 与 Qwen3.8-Flash-Next 两条模型线，包含完整项目包、生产二进制、可直接检查的生产配置、校验文件和来源记录。本版本冻结 DeepSeek-V4.1 的 R001–R179 生产优化结论，同时保留 Qwen3.8-Flash-Next 在 v0.2.3 已验证的引擎与运行方式。

## 1. 版本要点

- **DeepSeek-V4.1-Flash**：提供 EXL3 TP1、EXL3 TP4、GGUF TP2 三套生产配置，统一使用同一个生产二进制 `strata-dsv41`。
- **Qwen3.8-Flash-Next**：保留 `0.1.47` 默认快速引擎与 `0.1.55` 对照引擎，配置与启动脚本继续可用。
- **服务接口**：提供 OpenAI 兼容与 Anthropic 兼容 API，现有客户端无需改造即可接入。
- **质量门禁**：三套 DeepSeek 生产配置均通过 PPL 与固定 API 测试；结果与限制在本文中完整列出。
- **可校验发布**：所有关键资产提供 SHA-256 校验值，仓库 tag 内也包含完整 `SHA256SUMS`。

## 2. 运行能力

发布包内提供的服务接口如下：

```text
POST /v1/chat/completions    OpenAI 兼容
POST /v1/messages            Anthropic 兼容
GET  /v1/models
GET  /health
```

服务模型名为：

```text
deepseek-v4.1-flash
qwen3.8-flash-next
```

同一时间只建议运行一套模型服务。切换模型或配置前，应停止当前服务并重新检查设备占用。

## 3. 性能测试口径

- **TTFT**：指 API 请求提交到收到首个输出 token 的端到端时间，数值越低越好。
- **Prefill throughput**：按 prompt tokens 除以 prefill/TTFT 时间计算，数值越高越好。
- **Decode throughput**：指开始输出后每秒生成的 token 数，数值越高越好。
- **DeepSeek 测试输入**：4K 与 32K 两组 prompt 集合。
- **PPL 评测**：EXL3 两组覆盖 19,999 positions；GGUF TP2 使用 window 1024。质量门禁为最终 PPL 相对基线不超过 `+2%`，并额外执行固定四题 API 测试。

不同 prompt 内容、draft 命中率、系统负载和冷/热启动都会影响实测值。下表为发布前的验证结果，不作为对任意请求的固定性能承诺。

## 4. DeepSeek-V4.1-Flash 成绩

### 4.1 生产配置实测

| 配置 | 硬件拓扑 | 模型格式 | Prefill chunk | 4K TTFT | 32K TTFT | Decode 模式 | Decode 结果 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `exl3-tp1-prod.json` | HCU0–3，TP1 | EXL3 2.9 bpw | 1024 | 2.982 s | 17.953 s | Native | 18.7 tok/s |
| `exl3-tp4-prod.json` | HCU0–3，TP4 | EXL3 2.9 bpw | 3072 | 4.042 s | 29.220 s | DSpark | 43.9 tok/s |
| `gguf-tp2-prod.json` | HCU0–1，TP2 | GGUF 2.00 bpw | 512 | 8.110 s | 52.415 s | DSpark | 42.05 tok/s，中位数 |

补充说明：

- EXL3 TP1 的 4K TTFT 使用重复 Decode 批量中位数。R179 冷启动 A/B 中位为 3.114 s；生产 postinstall smoke 为 3.142 s。发布表采用可重复性更好的 2.982 s。
- EXL3 TP4 的 4K/32K 为生产安装后的 smoke 结果。
- GGUF TP2 的 DSpark 四组样本为 `37.3 / 36.4 / 46.8 / 46.9 tok/s`，呈双峰波动，因此只记录中位数 42.05 tok/s，不应解读为固定速度。
- GGUF TP2 的 Native Decode 对照结果为 22.4 tok/s。

### 4.2 自适应 Prefill A/B

R178/R179 引入有界自适应 Prefill。其核心规则为：非尾部、非最终 chunk 在短 prompt 与长 prompt 分别使用不同 active-K；最后 1024 rows 与最终 chunk 恢复完整 K；QSA candidate top-k=1024 只在达到行数阈值的请求中启用；Decode、DSpark draft 与 verify 不进入该近似 Prefill 路径。

#### EXL3 TP1

| 输入规模 | Off 中位 | On 中位 | 变化 |
| --- | ---: | ---: | ---: |
| 4K | 3.090 s | 3.114 s | 0.78%，冷启动噪声内 |
| 32K | 18.629 s | **17.953 s** | **−3.63%** |

#### EXL3 TP4

| 输入规模 | Off 中位 | On 中位 | 变化 |
| --- | ---: | ---: | ---: |
| 4K | 4.094 s | 4.070 s | −0.59% |
| 32K | 30.321 s | **29.352 s** | **−3.20%** |

#### GGUF TP2

| 指标 | Off | On | 变化 |
| --- | ---: | ---: | ---: |
| 4K TTFT | 8.604 s | **8.110 s** | **−5.75%** |
| 32K TTFT | 65.228 s | **52.415 s** | **−19.64%** |
| Native Decode | 22.5 tok/s | 22.4 tok/s | −0.44%，噪声内 |

### 4.3 质量评测

| 配置 | PPL 窗口 | Baseline PPL | Final PPL | 变化 | Top-1 | 固定 API |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GGUF TP2 | window 1024 | 1.741926 | 1.742965 | +0.060% | 0.895195 | 4/4 |
| EXL3 TP1 | window 1024 | 4.860520 | 4.847801 | −0.262% | 0.777489 | 4/4 |
| EXL3 TP4 | window 1 | 1.598806 | 1.601385 | +0.161% | 0.908845 | 4/4 |

三套配置均通过 `+2%` PPL 门禁和固定 API 测试。

## 5. Qwen3.8-Flash-Next 成绩

Qwen 两个二进制未重新修改，沿用 v0.2.3 已验证结果。发布包保留两个引擎，便于快速档与对照档切换。

| Prompt 规模 | `0.1.47` 快速引擎 | `0.1.55` 对照引擎 |
| ---: | ---: | ---: |
| 8,190 tokens | 4.375 s，1,846.2 tok/s | 7.860 s，1,033.6 tok/s |
| 40,057 tokens | 19.066 s，2,086.1 tok/s | 34.361 s，1,161.3 tok/s |

补充指标：

- `0.1.47` 在 8,191-token 输入下的 Decode 为 79.37 tok/s。
- `0.1.47` 的 MTP 接受结果为 `56 / 57`。
- 快速档允许数值漂移，不保证与对照档逐位一致。

## 6. 配置选择建议

| 部署目标 | 建议配置 |
| --- | --- |
| 四卡可用，优先 4K 请求首字延迟 | `configs/exl3-tp1-prod.json` |
| 四卡可用，优先持续生成速度 | `configs/exl3-tp4-prod.json` |
| 只有双卡，或已有 GGUF 2.00 bpw 资产 | `configs/gguf-tp2-prod.json` |
| 单卡 Qwen，优先 prompt 吞吐 | `scripts/k100ai/serve-k100ai.sh qwen` |
| 需要与 Qwen 快速档对照 | 设置 `K100AI_QWEN_ENGINE=bin/strata-0.1.55` |

EXL3 两套配置使用 HCU0–3；GGUF TP2 使用 HCU0–1。配置中的 `tokenizer`、`--pack`、`--expert-profile` 与日志路径必须按本机资产位置调整。

## 7. 下载与校验

推荐下载完整发布包：

```bash
BASE=https://github.com/DocPang/strata-k100ai/releases/download/v0.3.0

curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz"
curl -fLO "$BASE/strata-k100ai-v0.3.0.tar.gz.sha256"
sha256sum -c strata-k100ai-v0.3.0.tar.gz.sha256

tar -xzf strata-k100ai-v0.3.0.tar.gz
cd strata-k100ai-v0.3.0
sha256sum -c SHA256SUMS
```

也可以检出对应 tag：

```bash
git clone --branch v0.3.0 --depth 1 \
  https://github.com/DocPang/strata-k100ai.git
cd strata-k100ai
sha256sum -c SHA256SUMS
```

## 8. Release 资产清单

| 文件 | 大小 | SHA-256 |
| --- | ---: | --- |
| `strata-k100ai-v0.3.0.tar.gz` | 55,337,075 B | `de49da05602dc5647a4e4b686f13953c48b302d10264aea257d3d51c1f993fb9` |
| `strata-k100ai-v0.3.0.tar.gz.sha256` | 94 B | `44eee1ab0ef8cb00120458c859e1da3b2ad1fc435d5d87cfd38c85ddd1ecce9a` |
| `strata-dsv41` | 23,941,864 B | `4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6` |
| `strata-0.1.47` | 45,674,152 B | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `strata-0.1.55` | 49,496,856 B | `25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e` |
| `exl3-tp1-prod.json` | 3,515 B | `45140d27e37ffd4b6409d7c3f8a9ca015760155508ec89a3328b080796a83685` |
| `exl3-tp4-prod.json` | 3,469 B | `e7992fd677c904d42c8a09d53e741340817a95da2314c28cf5b62cea834f7a21` |
| `gguf-tp2-prod.json` | 3,974 B | `8ea92b9800c4c269262fb50f808e5062b6ece06e4d79c6302b90e6247e395364` |
| `qwen-server.template.json` | 2,628 B | `b4deeec2498c9d4dcb8c83cbb496602cf9eadbba54c0eb93b195c692d9a8769e` |
| `SHA256SUMS` | 315,572 B | `d826da3aede68b2bd2ab404f8dd180df4c93aef2a9bb1b5bc57691ad158b9499` |
| `RELEASE_NOTES_V0.3.0.md` | 见资产 | 以本页对应资产为准 |

模型权重、tokenizer、DeepSeek expert profile 和 Qwen MTP runtime 不包含在发布资产中，需要按配置另行准备。

## 9. 安装与启动

### 9.1 安装运行依赖

```bash
bash scripts/k100ai/setup-runtime.sh
```

启动前检查 HCU 占用：

```bash
bash scripts/k100ai/preflight.sh
```

### 9.2 启动 Qwen3.8-Flash-Next

默认资产路径为：

```text
$K100AI_ASSETS_DIR/qwen3.8-flash-next-iq3-xxs
```

如资产位于其他目录，可设置 `K100AI_QWEN_PACK`、`K100AI_QWEN_NATIVE`、`K100AI_QWEN_PLE`、`K100AI_QWEN_TOKENIZER`、`K100AI_QWEN_MTP_RT` 覆盖。

```bash
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

切换到 `0.1.55` 对照引擎：

```bash
K100AI_QWEN_ENGINE=bin/strata-0.1.55 \
K100AI_ASSETS_DIR=/data/k100ai-assets \
PORT=18096 \
  bash scripts/k100ai/serve-k100ai.sh qwen
```

### 9.3 启动 DeepSeek-V4.1-Flash

先检查并修改配置中的本机资产路径，然后启动：

```bash
bash scripts/k100ai/start-dsv41.sh configs/exl3-tp4-prod.json
```

DeepSeek 默认服务地址：

```text
http://127.0.0.1:18097
```

## 10. 接口冒烟测试

Qwen：

```bash
curl --fail http://127.0.0.1:18096/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

DeepSeek：

```bash
curl --fail http://127.0.0.1:18097/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

## 11. 二进制与来源

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `bin/strata-dsv41` | DeepSeek-V4.1 生产引擎 | `4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6` |
| `bin/strata-0.1.47` | Qwen Next 默认快速引擎 | `bdbfd934cf605074213c78529ba3f01236381ef2252ad562ade6369d6f4056f9` |
| `bin/strata-0.1.55` | Qwen Next 对照引擎 | `25d4ea83b883443d0a8bc68686c72f613cd0923a2897c2f7e9063939025e904e` |

DeepSeek 生产源码已与生产安装目录完成逐文件对账；关键源码哈希与构建环境记录见 `docs/k100ai/PROVENANCE_V0.3.md`。构建与验证环境记录为 `linux-x86_64-gfx928-dtk2604`。

## 12. 已知边界

- EXL3 TP1 的 32K TTFT 为 17.953 s，未达到 `<14.5 s` 目标。
- EXL3 TP4 的 32K TTFT 为 29.220/29.352 s，未达到 `<24 s` 目标。
- DSpark Decode 存在 run-to-run 波动，中位数不代表单次请求固定速度。
- 自适应 Prefill 是显式配置启用的有界近似路径，仅对本文列出的三套配置完成质量验证；其他模型、量化格式或拓扑需要重新执行 PPL、API、4K/32K 与 Decode A/B。
- 1M context 可在配置中设置，但完整 1M 窗口尚未完成质量评测。
- 发布机未重新编译出与 `strata-dsv41` 相同 SHA-256 的二进制；发布资产使用经过 R178 clean rebuild 与 R179 postinstall 验证的生产二进制，来源以逐文件源码对账和生产记录为准。

## 13. 相关文档

- [安装与启动](https://github.com/DocPang/strata-k100ai/blob/v0.3.0/docs/k100ai/INSTALL_V0.3.md)
- [来源与校验](https://github.com/DocPang/strata-k100ai/blob/v0.3.0/docs/k100ai/PROVENANCE_V0.3.md)
- [DeepSeek R001–R179 最终优化报告](https://github.com/DocPang/strata-k100ai/blob/v0.3.0/docs/k100ai/DSV41_OPTIMIZATION_FINAL.md)
- [模型运行参数与配置说明](https://github.com/DocPang/strata-k100ai/blob/v0.3.0/configs/README.md)
