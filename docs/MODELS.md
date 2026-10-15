# 模型、native pack 与 MTP

## 权重

本版本提供 Qwen3.8-Flash-Next GSQ-RCO 原版的 IQ3_S、IQ3_XXS 配置。模型来源及量化说明见 [固定版本上游模型文档](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/MODELS.md)。两个分片、tokenizer、pack 和 MTP 必须保持匹配。

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
  --env-file "$PWD/engines/strata/c4/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/models/next-iq3s/IQ3_S,dst=/models,readonly \
  --mount type=bind,src=/srv/packs,dst=/out \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/iq_pack.py \
  --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --out /out/next-iq3s --experts-bin
```

XXS 转换使用相同工具，修改模型挂载目录、第一分片文件名及输出路径：

```text
/srv/models/next-iq3xxs/IQ3_XXS → /models
--gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
--out /out/next-iq3xxs
```

转换需要足够的磁盘空间，输出目录与输入权重分开保存。

## MTP

目标 GGUF 不包含完整草稿头。可复用已核验的匹配 MTP runtime；重新准备时分为下载、验证、量化和运行时转换。以下下载步骤需要联网，其余步骤可增加 `--network none`。

```bash
mkdir -p /srv/mtp-work

docker run --rm -e STRATA_SOURCE=modelscope \
  --env-file "$PWD/engines/strata/c4/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/mtp-work,dst=/out \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" tools/mtp_fetch.py fetch --out /out/bf16
```

保留同样的挂载和环境，依次替换命令部分：

```bash
python3 tools/mtp_fetch.py verify --out /out/bf16
python3 tools/mtp_pack.py --src /out/bf16 --experts q2_0 --out /out/mtp-q2_0.gguf
python3 tools/mtp_rt.py --gguf /out/mtp-q2_0.gguf --out /out/runtime
```

上述三行为容器内命令。使用 `--entrypoint python3` 时，镜像名后从 `tools/...` 开始填写，不能再重复添加 `python3`。生成后将 `/srv/mtp-work/runtime` 挂载到推理容器 `/mtp`。

## XXS 格式覆盖

相对于 IQ3_S，XXS 的 gate/up 增加 IQ2_XS、IQ2_XXS 类型，down 中 Q2_0 的比例上升，目标输出头由 Q6_K 变为 Q5_K。c4 的通用路径支持这些格式，但并非所有张量均进入 IQ3_S 专用快路径。详见 [OPTIMIZATIONS.md](OPTIMIZATIONS.md)。

本版本增加 XXS 的 c4 基线配置，不包含另一份 `ad743dbf…` 实验引擎及其专用改动。两套权重的质量差异与实现正确性应分别评估。
