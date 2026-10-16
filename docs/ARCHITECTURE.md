# 项目结构

```text
engines/strata/xxs-20261016-gold/  当前引擎与实测记录（rc.6）
engines/strata/xxs-20261016/       rc.5 引擎、环境、源码增量与构建记录
engines/strata/c4/                 历史基线和基础源码恢复配方
profiles/next-iq3s-single/          IQ3_S 历史配置
profiles/next-iq3xxs-single/        IQ3_XXS 单卡配置
containers/strata/                 可选派生镜像与辅助服务入口
scripts/                          维护与验证工具
docs/                             部署、参数、容量规划与验证记录
```

## 配置接口

每个 profile 包含：

- `strata.json`：原生服务配置，直接交给 `python3 -m serve.server --config`。
- `profile.json`：辅助入口使用的路径模板与优化环境。
- `settings.json`：辅助入口的容量与 MTP 参数覆盖。
- `deployment.env.example`：辅助入口的宿主路径和设备模板。

推荐 XXS 原生部署的环境来自 `engines/strata/xxs-20261016-gold/environment.env`。原生配置与辅助配置独立读取，修改其中一种不会自动更新另一种。

## 版本管理

项目标签固定部署文件与配置组合；当前 `strata/xxs-20261016-gold`、上一版 `strata/xxs-20261016` 和历史 `strata/c4` 分别固定各自二进制身份与校验值，源码恢复记录目前只覆盖到 `strata/xxs-20261016` 快照；模型以独立来源和文件校验值固定。历史 Release 资产保留原始内容。

新增量化配置需记录模型来源、pack、格式覆盖、数值策略和运行验证。复用引擎不意味着内核命中比例或性能相同。

## TP2 扩展

TP2 将提供独立的设备列表、通信依赖、启动命令和验证记录。若使用不同推理后端，新增对应的 `engines/<backend>/<version>/` 与 `containers/<backend>/`。两个单卡副本按独立实例管理，不作为 TP2 配置。

Git 保存源码、文档、配置和补丁；Release 保存部署包与运行包；模型权重和 native pack 独立存储。
