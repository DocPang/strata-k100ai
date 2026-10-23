# 模型、native pack 与 MTP

## 权重

本版本推荐 Qwen3.8-Flash-Next GSQ-RCO 原版 IQ3_XXS；IQ3_S 模板作为历史资料保留。模型来源及量化说明见 [固定版本上游模型文档](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/MODELS.md)。两个分片、tokenizer、pack 和 MTP 必须保持匹配。

运行包不包含模型。第一分片用于主模型，第二分片用于 PLE mmap。下载后按发布者校验值核验；IQ3_XXS 已核对文件的 SHA-256 另存于 [weights.sha256](../profiles/next-iq3xxs-single/weights.sha256)。

```bash
# 在包含两个 XXS 分片的目录中执行，校验表路径按项目位置填写。
sha256sum -c /absolute/path/strata-k100ai/profiles/next-iq3xxs-single/weights.sha256
```

## native pack

不同量化配置分别生成 pack，不复用另一套权重的专家数据、偏移或类型表。`pack/tokenizer` 应存在。以下命令使用官方 DTK 镜像内的 Python，宿主无需 Python，也不需要映射 GPU。

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
mkdir -p /srv/packs

docker run --rm --network none \
  --env-file "$PWD/engines/strata/xxs-20261023-k100ai022/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/models/next-iq3xxs/IQ3_XXS,dst=/models,readonly \
  --mount type=bind,src=/srv/packs,dst=/out \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/iq_pack.py \
  --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --out /out/next-iq3xxs --experts-bin
```

转换需要足够的磁盘空间，输出目录与输入权重分开保存。

## MTP

目标 GGUF 不包含完整草稿头。可复用已核验的匹配 MTP runtime；重新准备时分为下载、验证、量化和运行时转换。以下下载步骤需要联网，其余步骤可增加 `--network none`。

```bash
mkdir -p /srv/mtp-work

docker run --rm -e STRATA_SOURCE=modelscope \
  --env-file "$PWD/engines/strata/xxs-20261023-k100ai022/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/mtp-work,dst=/out \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_fetch.py fetch --out /out/bf16
```

完整的四条 Docker 命令见[首页 MTP 步骤](../README.md#7-从魔塔准备-mtp-草稿模型)。以下为相同挂载下的容器内命令：

```bash
python3 tools/mtp_fetch.py verify --out /out/bf16
python3 tools/mtp_pack.py --src /out/bf16 --experts q2_0 --out /out/mtp-q2_0.gguf
python3 tools/mtp_rt.py --gguf /out/mtp-q2_0.gguf --out /out/runtime
```

上述三行为容器内命令。使用 `--entrypoint python3` 时，镜像名后从 `tools/...` 开始填写，不能再重复添加 `python3`。生成后将 `/srv/mtp-work/runtime` 挂载到推理容器 `/mtp`。

## 格式与质量

本版增加 IQ2_XS/IQ2_XXS 的 exact gate/up 路径，使该 XXS 权重的 48 层 gate/up 全部具备 exact 路径。selection 分数使用 FP16 缓冲，已测输出对照范围见 [BENCHMARKS.md](BENCHMARKS.md)。该策略不等于恢复量化前权重精度。

下载入口见 [魔塔模型仓库](https://modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/files)，完整直链与断点续传见 [INSTALL.md](INSTALL.md)。MTP 由 [魔塔 Qwen 官方模型](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/files)按需提取。

不同量化权重的输出可能不同。相同 XXS 权重下的新旧引擎输出对照与跨权重量化质量对照分别记录，见 [参考成绩](BENCHMARKS.md)。
