# 发布验证

## 固定程序与测试来源

v0.1.0-rc.6 分发 `strata-xxs-20261016-gold-linux-amd64.tar.gz`，包含以下冻结二进制：

```text
43002a7262e50d836f01642d1a5846565d8cd0734ec1d0ef67f534aeb11aafac
```

该文件为 2026-10-16 20:20（UTC+08:00）的增量构建，大小 29290960 字节，当天已部署到生产实例；发布时直接复制并校验原文件，不重新编译为另一个程序。上一版 v0.1.0-rc.5 分发 `d87a4678a21b`，其摘要、成绩与检查记录保留在 [rc.5 引擎目录](../engines/strata/xxs-20261016/) 供对照。

## 已有 GPU 记录

- rc.5 的 32K/128K/250K 同批 A/B/A prefill、250K 第二批重复测试、prefill=20480 对照，以及 MMVQ LDS 的 32K 输入 / 128-token 输出 A/B/A/B decode 检查，原始字段见 [rc.5 样本汇总](../engines/strata/xxs-20261016/benchmarks.json)。
- rc.6 只复测 32K 与 250K 单臂，并与上一版记录并列展示，未重做同批参考臂；跨版本百分数只是指示值，不是同批 A/B。
- rc.6 打包时对 HCU0 上正在运行的生产引擎（同一二进制）发起验证请求，原始字段见 [rc.6 样本汇总](../engines/strata/xxs-20261016-gold/benchmarks.json)。

两次生产验证请求先后执行，第二次的专家缓存已被第一次预热，不是受控 A/B；墙钟时间含客户端开销，不是引擎内部相位计时。250K 已测对照的输出 SHA 一致；短输入有一个 EOS 后 token 差异记录。FP16 scores 改变中间分数精度，不构成所有提示的位等价保证。已有测试不是多任务质量评测或长期浸泡测试。

## 发布打包检查

- 从 Release 重新下载 `strata-xxs-20261016-gold-linux-amd64.tar.gz`：8934284 字节，SHA-256 `241655d69476df9fb4159589bb7399d95daeba190c882544016f16b10d4b3155`，与 [artifact.sha256](../engines/strata/xxs-20261016-gold/artifact.sha256) 一致。
- 归档共 224 个条目；解包后 `runtime/FILES.sha256` 列出的 205 个文件全部校验通过。
- 解包得到的 `runtime/bin/strata` SHA-256 等于上面的发布摘要，大小 29290960 字节。
- 运行包继续使用已审核的 c4 Python 服务、转换工具和辅助数据，只有 `bin/strata` 相对 rc.5 不同。rc.5 的 15 项本地配置/离线安装测试、派生镜像构建、动态库解析、工具帮助入口与原生 lazy 服务检查记录见 [rc.5 发布检查记录](../engines/strata/xxs-20261016/release-validation.json)；这些检查本轮没有在 rc.6 上重跑。
- 仓库本地检查在本次文档更新后运行：`bash -n install.sh start.sh stop.sh scripts/config.sh` 通过；`python3 -m unittest discover -s tests` 15 项全通过，其中离线安装契约用重新下载的 rc.6 运行包以 `--local-only --bundle` 执行；文档内 bash 代码块语法检查通过。Compose 校验需要 Docker Compose 插件，本轮未执行。

本次打包不启动新的 GPU 研究实验、不切换生产服务。

## 源码与未完成项

附加源码补丁根据保留的源码与变更前备份恢复，校验表见 [source-recovery.json](../engines/strata/xxs-20261016/source-recovery.json)。补丁可在已恢复 c4 源码上回放；这不等于验证历史继承对象全部来自同一源码快照，也不等于完成干净重建。**rc.6 相对该快照的增量源码不在本仓库内**，见 [rc.6 引擎清单](../engines/strata/xxs-20261016-gold/manifest.json)。

仍需完成：完整干净源码重建、通用质量对照、该服务配置的 GPU 端到端与长时间稳定性验证、其他驱动/DTK 组合和 TP2 的兼容矩阵。公开默认不加载控制向量。
