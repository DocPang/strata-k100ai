# K100AI 配置指南

[项目首页](../../README.md) · [安装指南](../../docs/k100ai/INSTALL.md) · [运行指南](../../docs/k100ai/RUNTIME.md)

下表按每张 K100AI 64 GiB 列出 v0.3.2 发布包支持的配置。

| 卡数 | 配置 | 量化和解码 |
| ---: | --- | --- |
| 1 | [Qwen IQ3_S](qwen-iq3s-r112.template.json) | IQ3_S，目标模型验证 MTP |
| 2 | [DeepSeek GGUF TP2](gguf-tp2-prod.json) | GGUF 2.00 bpw，DSpark |
| 4 | [DeepSeek EXL3 TP1](exl3-tp1-prod.json) | EXL3 2.9 bpw，原生解码 |
| 4 | [DeepSeek EXL3 TP4](exl3-tp4-prod.json) | EXL3 2.9 bpw，DSpark |

## Qwen3.8-Flash-Next

`bash scripts/k100ai/serve-k100ai.sh qwen` 使用 IQ3_S 模板，启动
`bin/strata-qwen-iq3s-r112`。模板将上下文设为 262,144 tokens，KV 驻留设为 65,536 tokens，
保留 top-10 专家路由，关闭 prompt-lookup 后缀投机。资产路径、参数和实测见
[Qwen v0.3.1 指南](../../docs/k100ai/QWEN31.md)。

仍在使用 IQ3_XXS 资产时，用 `qwen-iq3xxs` 启动
[旧模板](qwen-server.template.json)。0.1.47 和 0.1.55 的切换方法见
[安装指南](../../docs/k100ai/INSTALL.md#qwen38-flash-next)。

## DeepSeek-V4.1-Flash

三套配置沿用 v0.3.0 的 `bin/strata-dsv41`。启动前根据部署位置修改
`tokenizer`、`log`、`--pack` 和配置中的 `--expert-profile`。EXL3 使用
HCU0–3，GGUF TP2 使用 HCU0–1。配置顶层的中文说明和哈希字段供查阅，
服务端只向引擎传递运行参数。

按 [DeepSeek 安装步骤](../../docs/k100ai/INSTALL.md#deepseek-v41-flash)
启动所选配置。切换前停止当前服务，并用 `bash scripts/k100ai/preflight.sh`
检查设备占用。二进制哈希、计时和 PPL 见
[DeepSeek 指南](../../docs/k100ai/DEEPSEEK41.md)。
