# 模型、native pack 与 MTP

本 Release 不包含 GGUF、native pack 或 MTP 权重，也不自动下载模型。首版配置针对 **Qwen3.8-Flash-Next GSQ-RCO 原版 IQ3_S**，不等于任何同名 IQ3_S 模型都兼容。

模型来源与上游说明见 [Strata 的固定版本模型文档](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/MODELS.md)。下载原发布者的两个 GGUF 分片并按其校验值核验，保留来源和哈希。分片一用于主模型，分片二用于 PLE mmap；第二分片不整体常驻显存。

## 已有 pack

复用时确认 pack、tokenizer、GGUF 两个分片和 MTP 运行时相互匹配。`PACK_DIR` 内必须有 `tokenizer`；只改权重路径、不重新生成对应 pack 不算换模型完成。

## 转换 native pack

安装优化镜像后，可在容器内使用工具，宿主无需 Python。以下转换命令不映射 GPU，输出目录需要有足够空间；将示例路径和文件名替换为自己的：

```bash
docker run --rm --network none \
  -v /srv/models/next-iq3s:/models:ro \
  -v /srv/packs:/out \
  k100ai/strata-c4:v0.1.0-rc.2 \
  tool iq_pack.py --gguf /models/model-00001-of-00002.gguf \
  --out /out/next-iq3s --experts-bin
```

## 独立 MTP

目标 GGUF 不包含完整 MTP 草稿头。已有校验过的 MTP runtime 可以直接挂载；否则执行独立下载、核验、转换。下载命令需要联网，其余步骤可离线：

```bash
mkdir -p /srv/mtp-work
docker run --rm -e STRATA_SOURCE=modelscope \
  -v /srv/mtp-work:/out k100ai/strata-c4:v0.1.0-rc.2 \
  tool mtp_fetch.py fetch --out /out/bf16

docker run --rm --network none \
  -v /srv/mtp-work:/out k100ai/strata-c4:v0.1.0-rc.2 \
  tool mtp_fetch.py verify --out /out/bf16

docker run --rm --network none \
  -v /srv/mtp-work:/out k100ai/strata-c4:v0.1.0-rc.2 \
  tool mtp_pack.py --src /out/bf16 --experts q2_0 --out /out/mtp-q2_0.gguf

docker run --rm --network none \
  -v /srv/mtp-work:/out k100ai/strata-c4:v0.1.0-rc.2 \
  tool mtp_rt.py --gguf /out/mtp-q2_0.gguf --out /out/runtime
```

最后设置 `MTP_DIR=/srv/mtp-work/runtime`。下载来源、权重许可和磁盘空间由用户自行检查。模型转换工具打包可调用不代表本次重新转换并验收了整套权重。

## 后续 IQ3_XXS

未来的 XXS 配置需独立记录权重哈希、pack、实际量化类型、内核覆盖和测试结果。当前 c4 历史上已被 XXS 服务复用，但本 Release 没有将未完成的 XXS 专用优化作为受支持配置提供。不要把 IQ3_S profile 直接改个文件名就发布为 XXS 优化版。
