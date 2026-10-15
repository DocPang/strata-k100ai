# 安装与基础环境

## 固定基础镜像

```text
harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
```

识别 tag：`0.5.12-ubuntu22.04-dtk2604-py3.10`。

Linux amd64 镜像配置 ID：`sha256:caeff0e9e6efef918a31359c4764b18bf2d428f2b76e3580c788f4cc8b0e93ad`。安装器检查 ID，所以该内容的本地别名可以使用；不支持用任意 DTK 镜像绕过版本约束。未来兼容其他基础版本，需要单独添加并验证。

基础镜像需要使用供应方允许的访问方式。若仓库要求登录，使用自己的凭据执行 `docker login harbor.sourcefind.cn:5443`。本项目不提供或收集该仓库凭据。

宿主机仍需匹配的 K100AI 驱动和 HyHAL。默认只读挂载宿主 `/opt/hyhal`，不在容器里安装宿主驱动。NVIDIA 原版 Strata 镜像不能替代本 DTK 环境。

## 安装器行为

1. 检查 Linux x86_64 和 Docker。
2. 查找基础镜像；缺少且允许联网时拉取。
3. 验证基础镜像内容 ID。
4. 从固定 Release 下载运行包，或读取 `--bundle` 指定文件。
5. 对照仓库内固定 SHA-256 校验，拒绝不匹配的包。
6. 从校验过的包创建临时构建目录，在基础镜像上 COPY 运行文件。
7. 使用 `--pull=false --network=none` 构建，检查包内文件校验与服务模块导入；没有 apt/pip 安装。
8. 生成本地镜像和私有 `deployment.env`。已有配置不覆盖，服务不自动启动。

如果装有 Buildx，安装器要求 `default` builder 使用 Docker driver，以便读取当前 Docker Engine 的本地镜像；不使用独立 docker-container 或远程 builder。基础镜像先建立本地别名，构建不再通过远端摘要解析 FROM。

## 离线准备

在有网络的同架构机器上取得官方镜像，根据供应方许可保存并搬运：

```bash
docker pull 'harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68'
docker tag 'harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68' my-local-dtk:2604
docker save -o dtk-base.tar my-local-dtk:2604
```

目标服务器执行 `docker load -i dtk-base.tar`。再传入本项目部署包和小运行包，按首页的 `--local-only` 命令安装。模型 GGUF、pack 和 MTP 同样单独搬运。

## 显卡、内存与启动

`ls -l /dev/dri/by-path/` 用于查看 PCI 路径与 render 节点对应关系。用供应方工具确认选定卡空闲。启动器只映射一个 render 节点及 `/dev/kfd`；引擎容器内使用 GPU 0。不要把全部显卡目录映射进去。

此配置使用全部 24,576 个专家槽，另有 dense、workspace、KV 与 MTP 占用。是否能容纳必须看引擎加载日志和实际空闲显存。宿主还需容纳 mmap/页缓存、KV pinned RAM 和最高 16 GiB 会话缓存；不保证低内存机器可用。

`seccomp=unconfined`、memlock unlimited 和宿主 HyHAL 挂载沿用当前 HIP 部署需求；启动前检查这些选项是否适合自己的宿主环境。容器默认不自动重启，避免加载失败反复占用显存。

默认 `start.sh` 直接调用 Docker Engine，无需 Compose。`deployment.env` 使用不带引号的字面量 `KEY=value`，不执行变量展开或命令。可选 Compose 启动：`docker compose --env-file deployment.env up -d`，停止时使用对应的 `docker compose ... down`；不要和默认启动器同时启动。

Compose v2 需支持长格式 bind 的 `create_host_path: false`；缺失模型目录会直接报错，而不是替用户生成空目录。

## 常见问题

- 校验失败：重新取得同一版本运行包，不编辑 SHA-256 绕过。
- 基础镜像不兼容：使用固定摘要对应内容；本地 tag 名字不限，内容必须匹配。
- 找不到 render 节点或权限错误：检查宿主驱动、设备和容器运行权限。
- 找不到 tokenizer：先完成对应 GGUF 的 native pack 转换。
- 模型与 pack 不一致：重新为当前权重转换；不要把 IQ3_S pack 用于 XXS。
- 显存不足：先确认没有其他任务占卡，降低 prefill。不要为追求启动擅自开启未验证开关。降低专家常驻量可能显著降低速度。
- 端口不可访问：默认仅宿主 `127.0.0.1`；远程访问需要明确调整 BIND_HOST 并使用 API key。

升级请使用新版本部署目录，保留原目录和本地镜像；先停止旧实例，再在同一张卡启动新实例。不要并行加载两个大模型进行“验证”。
