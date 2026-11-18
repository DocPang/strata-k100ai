# Qwen IQ3_S R112 发布证据

本目录保存 v0.3.1 发布说明所引用的原始数据。

## Native

`native/` 包含 4 次 32K / 256 原生 Decode 的配置、引擎日志和
`summary.json`。结果为 45.7–45.9 tok/s，input exact、output exact 和
retrieval 全部通过。

## MTP

`mtp/` 包含同一批请求启用 MTP 后的配置、引擎日志和 `summary.json`。
结果为 68.4–71.7 tok/s，四次中位 70.0 tok/s。每轮输出 SHA-256 与同轮
Native 完全一致。

## Prefill

`prefill-8k/` 和 `prefill-32k/` 各含 8 个内容不同、长度固定的 prompt。
`extracted.json` 给出可直接核对的逐次 Prefill 与 Decode 速率，嵌套目录
保留引擎日志和单次请求结果。

## 来源

生产二进制 SHA-256：

```text
102b4b412068a68588883c13a28103600d8b078785da9dcbbda34c440e9a5074
```

引擎最终使用 `--no-publish-kernel --suffix-draft 0`，发布模板显式设置
`STRATA_PREFILL_TOPK=10`。
