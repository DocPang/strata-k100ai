# Strata for K100AI

面向海光 K100AI（gfx928）的 Strata 推理优化项目。提供 GPU 内核适配、MoE 计算优化、显存管理、长上下文配置及可复现部署资料。

运行环境基于海光 DTK 镜像。推理引擎、服务程序和配置独立分发，模型权重与驱动由使用方准备。

## 版本与配置

当前版本：**v0.1.0-rc.3**。

| 配置 | 权重 | 引擎 | 拓扑 | MTP spec |
| --- | --- | --- | --- | ---: |
| [next-iq3s-single](profiles/next-iq3s-single/strata.json) | Qwen3.8-Flash-Next GSQ-RCO IQ3_S | Strata c4 | 单卡 | 4 |
| [next-iq3xxs-single](profiles/next-iq3xxs-single/strata.json) | Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS | Strata c4 | 单卡 | 6 |

两套配置均保留全部专家。XXS 使用 c4 基线及通用量化回退路径；XXS 专用实验内核不包含在本版本中。TP2 尚未发布。

运行包约 8.1 MiB。发布产物、基础镜像、源码及验证范围见 [版本说明](docs/VALIDATION.md)。

## 环境要求

- Linux x86_64、K100AI/gfx928，以及匹配的宿主驱动与 HyHAL。
- Docker Engine；原生启动不依赖 Compose 或宿主 Python 环境。
- 对应权重的两个 GGUF 分片、native pack、tokenizer 和独立 MTP runtime。
- 足够的显存、系统内存与磁盘空间。参考配置以专家全部驻留显存为目标，容量规划见 [显存与上下文](docs/MEMORY.md)。

## 获取运行文件

```bash
git clone https://github.com/DocPang/strata-k100ai.git
cd strata-k100ai
git checkout v0.1.0-rc.3

curl -fL --retry 3 \
  -o strata-c4-linux-amd64.tar.gz \
  https://github.com/DocPang/strata-k100ai/releases/download/v0.1.0-rc.3/strata-c4-linux-amd64.tar.gz
sha256sum -c engines/strata/c4/artifact.sha256
tar -xzf strata-c4-linux-amd64.tar.gz
(cd runtime && sha256sum -c FILES.sha256)
```

无 Git 环境可从 [Releases](https://github.com/DocPang/strata-k100ai/releases) 下载部署包。基础镜像的获取、本地镜像复用及离线准备见 [环境准备](docs/INSTALL.md)。

## 原生 Docker 启动示例

以下示例在指定的一张卡上启动 IQ3_S 服务。模型文件名由 `strata.json` 中的 `--native` 和 `--ple-gguf` 指定；模型目录、pack 和 MTP 使用宿主绝对路径。

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
RENDER_DEVICE=/dev/dri/renderD129
export STRATA_API_KEY="$(openssl rand -hex 24)"

docker run --rm -it --name strata-iq3s \
  --device /dev/kfd:/dev/kfd \
  --device "$RENDER_DEVICE:$RENDER_DEVICE" \
  --ulimit memlock=-1:-1 --shm-size=2g \
  --security-opt seccomp=unconfined \
  --tmpfs /var/log/hylog:rw,nosuid,nodev,mode=0755 \
  --publish 127.0.0.1:18641:8080 \
  --env STRATA_API_KEY \
  --env-file "$PWD/engines/strata/c4/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/models/next-iq3s/IQ3_S,dst=/models,readonly \
  --mount type=bind,src=/srv/packs/next-iq3s,dst=/pack,readonly \
  --mount type=bind,src=/srv/mtp-runtime,dst=/mtp,readonly \
  --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
  --mount "type=bind,src=$PWD/profiles/next-iq3s-single/strata.json,dst=/config/strata.json,readonly" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" -m serve.server --engine strata \
  --config /config/strata.json --host 0.0.0.0 --port 8080
```

`RENDER_DEVICE` 需按本机 PCI 与设备节点映射选择。宿主卡号不等于容器内 GPU 序号：示例只开放一个 render 节点，容器内使用 GPU 0。

完整的 **IQ3_S、IQ3_XXS 两套启动命令**、后台运行、日志、API 调用及直接引擎调用见 [运行与调试](docs/RUNNING.md)。服务启动命令直接调用上游服务模块，推理参数保存在可编辑的原生 `strata.json` 中。

## 参数与容量规划

| 参数 | IQ3_S | IQ3_XXS | 作用 |
| --- | ---: | ---: | --- |
| --expert-cache | 24576 | 24576 | 全部专家槽 |
| --prefill | 20480 | 20480 | 每次 prefill 的分块上限 |
| --max-context | 262144 | 262144 | 输入、历史与输出共用的上下文上限 |
| --kv | int8 | int8 | KV 存储格式 |
| --kv-resident | 32768 | 32768 | 显存 KV 驻留缓存容量 |
| --spec | 4 | 6 | MTP 草稿长度 |
| --vram-reserve-mib | 2048 | 2048 | 显存容量规划预留 |

IQ3_S 的专家原始数据约 46.84 GiB，IQ3_XXS 约 39.97 GiB；这些数值不含 dense、KV、MTP 与工作区。启用 KV streaming 后，增长上下文仍会增加系统内存及部分 GPU 状态开销。最大上下文参数不代表对应长度的时延和并发能力。

显存不足时优先检查其他进程占用及 prefill 工作区，再调整上下文与 KV 驻留。减少专家常驻量会增加数据搬运，可能显著降低解码速度。具体调节顺序与内存估算见 [MEMORY.md](docs/MEMORY.md)。

## 文档

- [环境准备](docs/INSTALL.md)
- [原生启动与调试](docs/RUNNING.md)
- [参数参考](docs/PARAMETERS.md)
- [显存、上下文与并发](docs/MEMORY.md)
- [模型、pack 与 MTP](docs/MODELS.md)
- [优化内容与格式覆盖](docs/OPTIMIZATIONS.md)
- [源码构建](docs/SOURCE_BUILD.md)
- [发布验证](docs/VALIDATION.md)
- [目录结构与后续扩展](docs/ARCHITECTURE.md)

## 来源与许可证

项目基于 [Strata](https://github.com/Niko1221/Strata) 和 [llama.cpp](https://github.com/ggml-org/llama.cpp)，保留 MIT 许可及原作者署名。本仓库部署代码使用 [MIT](LICENSE)，Strata 许可见 [engines/strata/LICENSE](engines/strata/LICENSE)。

本项目由独立维护者发布，与海光及 Strata 上游无官方隶属关系。DTK/HyHAL、模型权重及可选控制向量遵循各自的许可条件，不包含在运行包中。
