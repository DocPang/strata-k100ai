# v0.3.0 生产配置

根目录提供三套冻结的 DeepSeek-V4.1 配置，所有配置都使用同一个生产 clean
二进制：

```text
strata-dsv41
SHA-256 4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6
```

| 配置 | 模型格式 | 拓扑 | 用途 |
| --- | --- | --- | --- |
| `exl3-tp1-prod.json` | EXL3 2.9 bpw | 四卡 TP1 | 较低 4K 延迟，Native Decode |
| `exl3-tp4-prod.json` | EXL3 2.9 bpw | 四卡 TP4 | DSpark Decode |
| `gguf-tp2-prod.json` | GGUF 2.00 bpw | 双卡 TP2 | DSpark Decode，并采用 R178 自适应 Prefill |

配置中的 `description_zh`、`notes_zh`、`measured_zh`、`quality_zh`、
`binary_sha256` 和 `provenance_zh` 是发布元数据。服务端把配置读入 JSON
对象后只消费运行参数，未知元数据不会传给引擎，也不会改变启动参数。

## 启动前

1. `exe` 已指向 tag 内的 `bin/strata-dsv41`，`cwd` 已指向发布根目录。
2. 按实际机器修改 `tokenizer`、`log`、`--pack` 和 `--expert-profile`。
3. 确认模型资产和冻结 profile 存在。
4. 确认只使用 HCU0-3。EXL3 两套配置使用四张卡，GGUF TP2 使用 HCU0-1。
5. 校验二进制：

```bash
printf '%s  %s\n' \
  4fd967c1fa6769e60298357d55683c18e5522f6689441b12fe5a97489b3a35d6 \
  bin/strata-dsv41 | sha256sum -c -
```

## 启动

推荐使用发布包中的启动器：

```bash
bash scripts/k100ai/start-dsv41.sh configs/exl3-tp4-prod.json
```

同一时间只启动一套配置。切换配置前应停止当前服务并执行设备占用检查。

Qwen3.8-Flash-Next 继续使用 `configs/k100ai/qwen-server.template.json` 和
`scripts/k100ai/serve-k100ai.sh qwen`。

## 实测摘要

| 配置 | 4K TTFT | 32K TTFT | Decode |
| --- | ---: | ---: | ---: |
| `exl3-tp1-prod.json` | 2.98 s | 17.95 s | Native 18.7 tok/s |
| `exl3-tp4-prod.json` | 4.04 s | 29.22 s | DSpark 43.9 tok/s |
| `gguf-tp2-prod.json` | 8.11 s | 52.42 s | DSpark 42.05 tok/s，中位数 |

质量门和限制见
[`docs/k100ai/DSV41_OPTIMIZATION_FINAL.md`](../docs/k100ai/DSV41_OPTIMIZATION_FINAL.md)。
