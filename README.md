# K100AI Inference Kit

在已有海光 DTK 官方镜像上安装经过整理的推理优化程序。**本地有基础镜像就复用，没有就自动拉取；也支持完全使用本地文件安装。** 不需要在宿主机安装 Python，不在用户机器上重新编译 GPU 引擎。

首版：`v0.1.0-rc.1`，提供 **Strata / Qwen3.8-Flash-Next GSQ-RCO IQ3_S / K100AI 单卡** 配置。运行包约 **8.1 MiB**，基础镜像、模型 GGUF、native pack、MTP 权重、宿主驱动另外准备。

> 这是候选发布。安装包使用已核对 SHA-256 的历史 c4 引擎。源码补丁回放和 CMake 配置已验证，完整干净编译及新构建产物的 GPU 等价验证尚未完成。256K 是配置上限，不代表已保证 256K 延迟、并发或所有机器的稳定性。见 [验证范围](docs/VALIDATION.md)。

## 快速安装

需要 Linux x86_64、K100AI/gfx928、匹配的宿主驱动与 HyHAL、Docker Engine（默认启动不需要 Compose）。只选择空闲显卡。此配方以全部专家常驻显存为目标，不是通用低显存配方。

从 [Releases](https://github.com/DocPang/k100ai-inference-kit/releases) 下载 `deployment-v0.1.0-rc.1.tar.gz`，解压并进入目录：

```bash
tar -xzf deployment-v0.1.0-rc.1.tar.gz
cd k100ai-inference-kit-v0.1.0-rc.1
./install.sh
```

安装器自动取得小型运行包，校验其 SHA-256，复用/拉取固定的 DTK 基础镜像，生成本地镜像 `k100ai/strata-c4:v0.1.0-rc.1`。安装过程不启动服务、不改宿主驱动、不切换已有服务。基础环境完全没有时仍需要下载大型官方镜像。

编辑安装器生成的 `deployment.env`：

```dotenv
MODEL_DIR=/srv/models/next-iq3s
SHARD1=model-00001-of-00002.gguf
SHARD2=model-00002-of-00002.gguf
PACK_DIR=/srv/packs/next-iq3s
MTP_DIR=/srv/mtp-runtime
HYHAL_DIR=/opt/hyhal
RENDER_DEVICE=/dev/dri/renderD128
BIND_HOST=127.0.0.1
PORT=8080
STRATA_API_KEY=替换为至少16字符的私有密钥
```

`SHARD1/SHARD2` 是实际文件名；其他目录使用宿主机绝对路径。安装器在有 openssl 时自动生成私有密钥，已有配置不会覆盖。显卡节点请在本机确认，不照抄别人的卡号。模型准备方法见 [模型与 pack](docs/MODELS.md)。

```bash
./start.sh
docker logs -f k100ai-next-iq3s-single
curl --fail http://127.0.0.1:8080/health
```

`./start.sh --check` 可先检查路径和设备，不启动容器。已有同名容器时不会自动替换；修改配置后先 `./stop.sh`，再 `./start.sh`。

API 地址：`http://127.0.0.1:8080/v1`；API key 使用自己的 `STRATA_API_KEY`。默认只发布到宿主回环地址。需要远程访问时修改 `BIND_HOST`，保留 API key。停止服务：

```bash
./stop.sh
```

## 使用本地镜像 / 离线安装

用户已经有相同基础镜像内容时，可以指定本地 tag：

```bash
./install.sh --base-image my-local-dtk:2604
```

完全离线时，同时从 Release 下载 `strata-c4-linux-amd64.tar.gz`，把它和部署包传入服务器：

```bash
./install.sh --local-only \
  --base-image my-local-dtk:2604 \
  --bundle /absolute/path/strata-c4-linux-amd64.tar.gz
```

`--local-only` 缺少本地镜像或运行包就报错，不自动下载。只支持已验证基础镜像的相同内容，改个名字不能让不兼容镜像通过检查。见 [安装细节](docs/INSTALL.md)。

## 启动参数与优化内容

- [参数说明](docs/PARAMETERS.md)：prefill、上下文、专家缓存、MTP、KV、会话缓存和环境开关。
- [优化清单](docs/OPTIMIZATIONS.md)：平台移植、量化核、MoE、prefill、decode、内存与部署调优；区分源码改动和上游能力配置。
- [源码构建](docs/SOURCE_BUILD.md)：固定原版提交、补丁与文件校验；包含未完成的验证项。
- [版本验证](docs/VALIDATION.md)：安装和运行包的证据及边界。

## 项目结构与后续版本

```text
engines/<后端>/<引擎版本>/    运行产物身份、源码补丁、构建来源
profiles/<模型-量化-拓扑>/   单独管理模型类型、拓扑和参数
containers/<后端>/          对应服务入口与镜像配方
scripts/                    维护与打包工具
install.sh / compose.yaml   当前发行版安装和启动入口
docs/                       使用、优化、验证与扩展约定
```

当前可安装配置只有 `next-iq3s-single`。**IQ3_XXS 和 TP2 后续分别加入配置、引擎产物及验证记录**；不会只改一个名字就宣称支持。TP2 若使用不同推理后端，也可以放在新的 `engines/<backend>/` 和 `containers/<backend>/` 下，不与 Strata 混在一起。详情见 [扩展约定](docs/ARCHITECTURE.md)。

## 来源与许可证

本项目是独立的 K100AI 部署与优化整理项目，不是海光或 Strata 官方发行版。

- 上游 [Niko1221/Strata](https://github.com/Niko1221/Strata)，固定参考 `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`，MIT。
- 底层依赖 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)，固定参考 `3cf03257f219afbe7334045ff7c6a06ac68c627d`，MIT。
- 本仓库部署代码使用 [MIT](LICENSE)，保留 [Strata 许可](engines/strata/LICENSE)；运行包包含上游许可文件。
- 基础镜像由用户从供应方取得；本项目不随包分发 DTK/HyHAL 库、模型权重或控制向量，各自适用其许可证。
