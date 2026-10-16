# 环境准备

从权重下载到启动的完整顺序见[首页安装教程](../README.md#安装与运行)。本节补充本地镜像复用、离线传输和手动派生镜像构建。

## 基础镜像

参考环境为 Linux amd64、Ubuntu 22.04、DTK 2604、Python 3.10：

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
docker pull "$BASE_IMAGE"
```

对应 tag 为 `0.5.12-ubuntu22.04-dtk2604-py3.10`。本地已有该镜像时，可直接将 `BASE_IMAGE` 设为本地 tag，不必重新下载：

```bash
BASE_IMAGE=my-local-dtk:2604
docker image inspect "$BASE_IMAGE" --format '{{.Id}}'
```

本版本使用的固定镜像配置 ID 为：

```text
sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad
```

更换基础镜像需要重新检查动态库、Python 依赖和模型加载。宿主机需安装兼容的 K100AI 驱动与 HyHAL；容器挂载宿主 `/opt/hyhal`。镜像访问需要认证时，由使用方执行 `docker login harbor.sourcefind.cn:5443`。

## 运行包与目录

在项目根目录下载并校验运行包：

```bash
curl -fL --retry 3 -o strata-xxs-20261016-linux-amd64.tar.gz \
  https://github.com/DocPang/strata-k100ai/releases/download/v0.1.0-rc.5/strata-xxs-20261016-linux-amd64.tar.gz
sha256sum -c engines/strata/xxs-20261016/artifact.sha256
tar -xzf strata-xxs-20261016-linux-amd64.tar.gz
(cd runtime && sha256sum -c FILES.sha256)
```

校验失败时停止部署。运行文件布局：

```text
runtime/
  bin/strata          C++/HIP 推理引擎
  app/serve/          HTTP 服务与协议处理
  app/tools/          模型转换工具
  app/data/           专家分布与词表等辅助数据
  app/gguf-py/        固定版本的 GGUF 工具依赖
  FILES.sha256        文件校验表
  PROVENANCE.json     产物来源
```

原生启动将该目录挂载到 DTK 容器中，不需要构建派生镜像。模型权重、pack 和 MTP 应放在独立目录，准备方式见 [MODELS.md](MODELS.md)。

## 显卡设备

```bash
ls -l /dev/dri/by-path/
ls -l /dev/dri/renderD*
```

根据 PCI 地址与供应方设备工具确认目标卡，检查是否已有任务占用。启动时映射 `/dev/kfd` 和目标 render 节点，并保留节点原路径。例如：

```bash
--device /dev/kfd:/dev/kfd \
--device /dev/dri/renderD129:/dev/dri/renderD129
```

不要将宿主 `renderD129` 重命名为容器 `renderD128`，也不要为单卡实例挂载整个 `/dev/dri`。参考环境中，单节点映射配合 `HIP_VISIBLE_DEVICES=0`、`ROCR_VISIBLE_DEVICES=0` 使用；需通过启动日志确认目标设备。

## 离线环境

在可联网机器下载部署包和运行包，并根据供应方许可导出基础镜像：

```bash
docker tag "$BASE_IMAGE" dtk-local:2604
docker save -o dtk-base.tar dtk-local:2604
```

目标服务器导入后使用本地名称：

```bash
docker load -i dtk-base.tar
BASE_IMAGE=dtk-local:2604
```

权重、pack、MTP runtime 与项目文件需同时传入。文件准备完成后，原生启动不需要访问模型仓库或 Python 包源。

## 可选派生镜像

需要将运行文件固定在派生镜像中时，可手动构建：

```bash
mkdir -p .cache
BUILD_DIR="$(mktemp -d "$PWD/.cache/manual-build.XXXXXX")"
cp -R runtime "$BUILD_DIR/runtime"
cp -R profiles "$BUILD_DIR/profiles"
cp containers/strata/Dockerfile "$BUILD_DIR/Dockerfile"
cp containers/strata/entrypoint.py "$BUILD_DIR/entrypoint.py"

docker tag "$BASE_IMAGE" k100ai-base-dtk:2604-caeff0e9
BUILDX_BUILDER=default docker build --pull=false --network=none \
  --build-arg BASE_IMAGE=k100ai-base-dtk:2604-caeff0e9 \
  --build-arg ENGINE_ID=strata/xxs-20261016 \
  -t k100ai/strata-xxs-20261016:v0.1.0-rc.5 "$BUILD_DIR"
```

采用 Docker-backed builder 复用本机镜像。Dockerfile 不执行 apt/pip 安装。仓库保留 `install.sh`、`start.sh` 和 Compose 作为兼容入口；主要运行与调试流程见 [RUNNING.md](RUNNING.md)。

## 魔塔权重下载

[魔塔模型页面](https://modelscope.cn/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/files)。以下为可断点续传的两个直接下载命令：

```bash
mkdir -p /srv/models/next-iq3xxs/IQ3_XXS
cd /srv/models/next-iq3xxs/IQ3_XXS
curl -fL --retry 3 -C - -o Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  'https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=IQ3_XXS%2FQwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf'
curl -fL --retry 3 -C - -o Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  'https://modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo?Revision=master&FilePath=IQ3_XXS%2FQwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf'
sha256sum -c /absolute/path/strata-k100ai/profiles/next-iq3xxs-single/weights.sha256
```

2026-10-16 核验：两个直链可访问，大小与 SHA-256 对应已核验权重。DTK 镜像来自海光国内 sourcefind 仓库，已核验匿名拉取的 manifest 摘要；未将其他平台的镜像链接作为其副本。
