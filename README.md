# Strata for K100AI

面向海光 K100AI（gfx928、64 GiB 显存）的 Qwen3.8-Flash-Next 推理优化。当前预发布版本 **v0.1.0-rc.4** 提供 **IQ3_XXS、exact+fused 引擎、256K 全 KV 显存驻留、单槽 FIFO**，多卡并发使用独立实例和网关分发。

本版为预发布版本，提供部署配置、运行程序和使用文档。发布附件中的引擎（SHA-256 前缀 `5d87f09e07bb`）已完成增量编译与打包检查，尚未完成该二进制的独立 GPU 质量与性能测试。

[参考成绩](docs/BENCHMARKS.md) 分别列出已测试的 XXS exact+fused 基准引擎（`ad743dbf9cc1`）和 2026-10-16 长上下文实验引擎（`ecd44f1c900c`）的结果。两者均不同于发布附件中的二进制；实验引擎的全部优化尚未纳入本运行包。

## 下载

| 文件 / 环境 | 国内来源或下载地址 |
| --- | --- |
| DTK 基础镜像 | [海光国内镜像仓库](https://harbor.sourcefind.cn:5443/)；下方提供固定摘要的 pull 命令 |
| IQ3_XXS 权重 | [魔塔模型仓库](https://modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/files) |
| GGUF 分片 1 | [魔塔直链，约 47.04 GB](https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=IQ3_XXS%2FQwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf) |
| GGUF 分片 2 | [魔塔直链，约 28.80 GB](https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=IQ3_XXS%2FQwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf) |
| MTP 来源 | [魔塔 Qwen 官方模型](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/files)，按需提取方法见 [模型准备](docs/MODELS.md) |
| 优化运行包与部署文件 | [GitHub Release](https://github.com/DocPang/strata-k100ai/releases/tag/v0.1.0-rc.4) |

DTK 镜像的已核验来源为海光国内仓库，权重由魔塔分发。两者是独立资源；未提供未经验证的魔塔 DTK 镜像副本。运行包约 8.1 MiB，不含驱动和模型。

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
docker pull "$BASE_IMAGE"
```

本地已有相同内容的镜像时可使用本地 tag。权重下载命令、SHA-256、离线准备见 [下载与环境](docs/INSTALL.md)。

## 推荐配置

| 项目 | 设置 |
| --- | --- |
| 权重 / pack | IQ3_XXS；独立生成对应 native pack |
| --expert-cache | 24576，全部专家驻留 |
| --max-context | 262144 |
| --kv | int8 |
| --kv-resident | 省略，完整 KV 驻留显存 |
| --prefill | 20480 |
| --spec / --spec-min-p | 6 / 0.70 |
| --vram-reserve-mib | 2048 |
| --batch / --batch-mtp | 均省略，单槽 FIFO |
| STRATA_PF_FUSED | 1 |
| STRATA_PF_FUSED_NATIVE_EXACT_SKIP | 1，29 层保持 exact，19 层 IQ2 走 fused |
| STRATA_SEL_BATCH | 1024，QSA selection 参数；不增加会话槽 |

发布配置显式设置 `SEL_BATCH=1024`。其已测收益来自另一个引擎的 128K 实验；收益随上下文和显存布局变化，不作为所有 256K 请求的速度保证。适用条件见 [参考成绩](docs/BENCHMARKS.md)。

## 原生启动

下载发布文件并解压运行包，校验后会得到 `runtime/`：

```bash
curl -fL --retry 3 -o strata-xxs-exact-sel-linux-amd64.tar.gz \
  https://github.com/DocPang/strata-k100ai/releases/download/v0.1.0-rc.4/strata-xxs-exact-sel-linux-amd64.tar.gz
sha256sum -c engines/strata/xxs-exact-sel/artifact.sha256
tar -xzf strata-xxs-exact-sel-linux-amd64.tar.gz
(cd runtime && sha256sum -c FILES.sha256)

RENDER_DEVICE=/dev/dri/renderD128
export STRATA_API_KEY="$(openssl rand -hex 24)"

docker run --rm -it --name strata-iq3xxs \
  --device /dev/kfd:/dev/kfd --device "$RENDER_DEVICE:$RENDER_DEVICE" \
  --ulimit memlock=-1:-1 --shm-size=2g \
  --security-opt seccomp=unconfined \
  --tmpfs /var/log/hylog:rw,nosuid,nodev,mode=0755 \
  --publish 127.0.0.1:18640:8080 --env STRATA_API_KEY \
  --env-file "$PWD/engines/strata/xxs-exact-sel/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/models/next-iq3xxs/IQ3_XXS,dst=/models,readonly \
  --mount type=bind,src=/srv/packs/next-iq3xxs,dst=/pack,readonly \
  --mount type=bind,src=/srv/mtp-runtime,dst=/mtp,readonly \
  --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
  --mount "type=bind,src=$PWD/profiles/next-iq3xxs-single/strata.json,dst=/config/strata.json,readonly" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" -m serve.server --engine strata \
  --config /config/strata.json --host 0.0.0.0 --port 8080
```

启动前替换模型、pack、MTP 路径，并确认目标卡空闲。服务参数位于原生 `strata.json`，优化环境位于 `environment.env`。选卡、后台运行、日志和直接引擎调用见 [运行与调试](docs/RUNNING.md)。

## 参考成绩

以下为同一张 K100AI 上的历史部署对照，XXS 使用 exact+fused 基准引擎（`ad743dbf9cc1`）：

| 指标 | IQ3_S | IQ3_XXS + exact/fused |
| --- | ---: | ---: |
| 加载显存（32K KV streaming 配置） | 60.71 GiB | 52.46 GiB |
| Decode，8192 入 / 512 出 | 62.5 tok/s | 60.3 tok/s |
| Prefill，8192 token | 1388.5 tok/s | 1412.9 tok/s |
| Prefill，131072 token | 1235.3 tok/s | 1239.9 tok/s |
| Prefill，261120 token | 745.5 tok/s | 731.7 tok/s |

XXS 全 KV 驻留历史实验记录为 decode 61.0 tok/s、加载后空闲 8186 MiB。XXS 成绩来自基准引擎（`ad743dbf9cc1`），不代表 rc.4 发布引擎（`5d87f09e07bb`）的独立测试结果。完整成绩、并发对照和正确性范围见 [BENCHMARKS.md](docs/BENCHMARKS.md)。

## 文档与发布范围

[环境准备](docs/INSTALL.md) · [模型准备](docs/MODELS.md) · [参数](docs/PARAMETERS.md) · [显存](docs/MEMORY.md) · [成绩](docs/BENCHMARKS.md) · [源码](docs/SOURCE_BUILD.md) · [验证](docs/VALIDATION.md)

新引擎以 XXS exact+fused 基准引擎（`ad743dbf9cc1`）为基础，仅回移植 selection batch 的三处宿主参数/显存记账，编译后的 SHA-256 为 `5d87f09e…`。已完成源码回放和增量编译链接；该二进制的独立 GPU 质量与性能测试尚未完成。IQ3_S/c4 配方作为历史资料保留，TP2 尚未发布。

项目基于 [Strata](https://github.com/Niko1221/Strata) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp)，遵循 [MIT](LICENSE) 并保留[上游许可](engines/strata/LICENSE)。本项目由独立维护者发布。DTK/HyHAL、权重和控制向量遵循各自许可。
