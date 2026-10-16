# exact+fused + selection batch

作为修改基础的 XXS exact+fused 基准引擎，其 SHA-256 为 `ad743dbf9cc15026758cfceb26e2dd0aeb25700b295fba028e64ca4b33bc18c0`。

发布引擎：`5d87f09e07bbe42579322b93e33f13e4f2916decd1c1a8450bc2651ae1e251a1`。复用上述基准引擎的其他静态对象，只重新编译 prefill.cpp，将 `STRATA_SEL_BATCH` 回移植到实际选择批次和两处配套显存记账。默认 256，合法覆盖 32–8192。推荐配方显式覆盖为 1024。

`patches/c4-to-exact-sel.patch` 包含从已恢复 c4 源码到本引擎 prefill.cpp 的累计差异（exact-skip + selection 参数），应用后哈希见 build-record.json。已校验 XXS 基准引擎与 c4 的 src/include 共 316 个输入，仅 prefill.cpp 不同；发布补丁回放后文件与编译输入一致。

增量编译和链接在固定 DTK 镜像中无 GPU、无网络完成。继承对象不等于完成干净全量重建，发布构建尚未独立完成 GPU 质量与性能验收。参考成绩逐项标注实际使用的引擎和测试配置。

源码恢复步骤见 [SOURCE_BUILD.md](../../../docs/SOURCE_BUILD.md)，工作区和参数说明见 [MEMORY.md](../../../docs/MEMORY.md)。
