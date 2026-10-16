# 发布验证

## 引擎身份

| 项 | SHA-256 / 状态 |
| --- | --- |
| c4 历史基线 | c4be512a7d257d9b56bc325136431e5feffd2b6d76f52378051829bfe73fc903 |
| XXS exact+fused 基准引擎 | ad743dbf9cc15026758cfceb26e2dd0aeb25700b295fba028e64ca4b33bc18c0 |
| 本版 exact+fused + SEL_BATCH | 5d87f09e07bbe42579322b93e33f13e4f2916decd1c1a8450bc2651ae1e251a1 |

`ad743dbf…` 不支持 SEL_BATCH；selection 参数实验引擎 `d0c643af96a9` 支持 SEL_BATCH，但不含 exact-skip。为同时提供两项功能，本版仅在 XXS 基准引擎的 prefill.cpp 中增加 selection 参数，并同步修改实际调度及两处显存记账，默认 256，发布环境覆盖为 1024。2026-10-16 长上下文实验引擎的其他内核优化未纳入本运行包。

## 已完成

- 校验 XXS exact+fused 基准引擎的 src/include 共 316 个文件：相对已恢复 c4 源码仅 prefill.cpp 不同。
- 从 c4 应用累计附加补丁后，prefill.cpp 与此次编译输入逐字节相同，哈希见 [构建记录](../engines/strata/xxs-exact-sel/build-record.json)。
- 在固定 DTK 镜像、无 GPU、无网络、CPU=2 的容器中重新编译 prefill.cpp，并使用 XXS 基准引擎的静态对象完成链接。
- 前端、工具和辅助数据沿用已逐文件核对的 174 个运行文件；运行包内所有文件附 SHA-256。
- 原生配置省略 kv-resident、batch 和 batch-mtp；辅助入口生成相同的全驻留单槽参数。
- 国内 DTK 镜像可匿名获取 manifest，摘要与固定镜像一致；魔塔 GGUF 直链与文件大小、SHA-256 清单匹配。

本版已完成本地发布检查（15 项测试，包括离线安装流程、配置生成、原生命令语法）及 Compose 配置检查。新增二进制尚未执行独立的动态库、help、派生镜像运行和 lazy 服务检查；rc.3 的历史检查不能替代本版验证。

## 历史运行证据

XXS exact+fused 基准引擎（`ad743dbf9cc1`）已完成 XXS 加载、四组新旧引擎输出 SHA、短请求、SSE、工具调用及缓存五项检查；长上下文、全 KV 和 batch 数据见 [参考成绩](BENCHMARKS.md)。

§186 的 SEL_BATCH=1024 正确性和耗时数据来自 selection 参数实验引擎 `d0c643af96a9` 的 ctx128 计时实验。§187 显示更大 selection 在不同上下文下存在收益反转，不能用该相位结果作为本版全驻留 XXS 的整模保证。

## 待完成

- 本次回移植构建的独立 GPU 数值、文本、质量及 A/B 性能验收。
- 完整干净重建；现有构建复用 XXS 基准引擎的静态对象，全量重建所需输入的来源校验尚未完成。
- 本版 256K 长时间稳定性及多卡网关实际负载验收。
- TP2 和其他 DTK/驱动组合的兼容矩阵。

公开默认不加载控制向量，采用与 XXS 基准引擎相同的 FP16 中间结果策略。历史质量小样本和输出 SHA 一致性不构成跨权重无损证明。
