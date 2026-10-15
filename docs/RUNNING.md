# 运行与调试

本页使用官方 DTK 基础镜像直接启动服务。示例均在项目根目录执行，运行包已解压到 `runtime/`。路径应替换为本机目录。

## 配置文件

| 权重 | 原生服务配置 | 模型挂载目录 | pack 挂载目录 | 宿主端口 |
| --- | --- | --- | --- | ---: |
| IQ3_S | profiles/next-iq3s-single/strata.json | /srv/models/next-iq3s/IQ3_S | /srv/packs/next-iq3s | 18641 |
| IQ3_XXS | profiles/next-iq3xxs-single/strata.json | /srv/models/next-iq3xxs/IQ3_XXS | /srv/packs/next-iq3xxs | 18640 |

`strata.json` 为服务模块原生配置，包含引擎路径、`args`、tokenizer 和后端。可直接编辑 `args` 调整参数，无需使用配置生成器。`env` 默认留空，继承容器的优化环境；在其中填写的变量会覆盖继承值。

共同环境保存在 `engines/strata/c4/environment.env`，包含 HIP 设备可见性、动态库路径和优化开关。该文件不包含 API key。

## IQ3_S 单卡参考命令

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

## IQ3_XXS 单卡参考命令

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
RENDER_DEVICE=/dev/dri/renderD128
export STRATA_API_KEY="$(openssl rand -hex 24)"

docker run --rm -it --name strata-iq3xxs \
  --device /dev/kfd:/dev/kfd \
  --device "$RENDER_DEVICE:$RENDER_DEVICE" \
  --ulimit memlock=-1:-1 --shm-size=2g \
  --security-opt seccomp=unconfined \
  --tmpfs /var/log/hylog:rw,nosuid,nodev,mode=0755 \
  --publish 127.0.0.1:18640:8080 \
  --env STRATA_API_KEY \
  --env-file "$PWD/engines/strata/c4/environment.env" \
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

两个示例分别使用不同显卡和端口，属于两个独立实例，不构成 TP2。设备节点仅为参考，启动前应确认对应显卡空闲。两份配置使用同一份 c4 运行包，但必须使用各自权重生成的 pack。

## Docker 参数

| 参数 | 作用 |
| --- | --- |
| --rm -it | 前台运行，直接查看日志；退出后删除容器 |
| --device | 开放 KFD 和选定 GPU 的 render 节点 |
| --ulimit memlock=-1:-1 | 允许运行时分配所需的锁页内存 |
| --shm-size=2g | 容器共享内存上限；不是显存配额 |
| --security-opt seccomp=unconfined | 使用参考 HIP 环境的系统调用配置 |
| --tmpfs /var/log/hylog:... | 为 HyHAL 提供容器内临时日志目录 |
| --publish 127.0.0.1:18641:8080 | 将容器 8080 映射到宿主回环地址的 18641 端口 |
| --env STRATA_API_KEY | 从宿主当前环境传入 API key，不在命令行中写入密钥值 |
| --env-file | 加载可审阅、可编辑的优化环境变量 |
| --mount ... readonly | 将运行文件、权重和配置以只读方式挂载 |
| --workdir /opt/k100ai/app | 设置服务模块的工作目录 |
| --entrypoint python3 | 直接运行 Strata 服务模块，绕过镜像默认入口 |

服务在容器中监听 `0.0.0.0:8080`，外部可达范围由 `--publish` 控制。需要对外开放时更改宿主绑定地址，并保留 API key。

## 调整参数

停止目标实例后编辑对应 `strata.json`，再用原命令重启。例如降低 prefill 工作区：

```json
"--prefill", "8192",
"--max-context", "262144",
"--kv", "int8",
"--kv-resident", "32768"
```

只需要 64K 上下文时，可将 `--max-context` 改为 `65536`。如需测试全量 KV 驻留，可将 `--kv-resident` 改为 `0`；应先按 [MEMORY.md](MEMORY.md) 核算额外显存。参数变化后重载模型，不能依靠编辑文件实现热更新。

参考部署使用 IQ3_S `spec=4`、IQ3_XXS `spec=6`，其余主要容量参数一致。公开配置默认不加载控制向量；复现带控制向量的部署需显式增加相应参数，见 [参数参考](PARAMETERS.md)。

## 后台运行、检查与停止

将启动命令中的 `--rm -it` 替换为 `--detach` 可保留容器并在后台运行。以下以 IQ3_S 为例：

```bash
docker logs -f strata-iq3s
docker exec -it strata-iq3s tail -f /tmp/strata-engine.log
curl --fail http://127.0.0.1:18641/health
curl --fail -H "Authorization: Bearer $STRATA_API_KEY" \
  http://127.0.0.1:18641/v1/models

docker stop strata-iq3s
docker rm strata-iq3s
```

前台 `--rm` 模式退出后不需要再次执行 `docker rm`。API 请求示例：

```bash
curl --fail http://127.0.0.1:18641/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-strata","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"stream":true}'
```

每个示例中的随机 key 仅保存在执行命令的 shell 环境。客户端应配置对应实例的 key；正式服务可通过权限受限的环境文件维护密钥。

## 直接调用 C++ 引擎

需要绕过 HTTP 前端分析加载、prefill 或 decode 时，可直接运行引擎。输入为与当前 tokenizer 匹配的 token ID 文件，不是 UTF-8 提示词。先将预分词结果保存在本机 `prompt.ids`。

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
RENDER_DEVICE=/dev/dri/renderD129

docker run --rm -it --name strata-iq3s-engine \
  --device /dev/kfd:/dev/kfd \
  --device "$RENDER_DEVICE:$RENDER_DEVICE" \
  --ulimit memlock=-1:-1 --shm-size=2g \
  --security-opt seccomp=unconfined \
  --tmpfs /var/log/hylog:rw,nosuid,nodev,mode=0755 \
  --env-file "$PWD/engines/strata/c4/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/models/next-iq3s/IQ3_S,dst=/models,readonly \
  --mount type=bind,src=/srv/packs/next-iq3s,dst=/pack,readonly \
  --mount type=bind,src=/srv/mtp-runtime,dst=/mtp,readonly \
  --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
  --mount "type=bind,src=$PWD/prompt.ids,dst=/input/prompt.ids,readonly" \
  --workdir /opt/k100ai/app --entrypoint /opt/k100ai/bin/strata \
  "$BASE_IMAGE" \
  --gpu 0 --pack /pack \
  --native /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --ple-gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf \
  --ple-io mmap --mmap-experts \
  --expert-cache 24576 --expert-profile /opt/k100ai/app/data/expert-profile.bin \
  --vram-reserve-mib 2048 --prefill 20480 \
  --max-context 262144 --kv int8 --kv-resident 32768 \
  --mtp /mtp --spec 4 --spec-min-p 0.70 --pool-workers 12 \
  --eos-ids 248046 --no-prefill-borrow \
  --tokens-file /input/prompt.ids --max-new 128
```

`BASE_IMAGE` 与 `RENDER_DEVICE` 应在该 shell 中重新确认，不能使用前一实例留下的错误设备值。该示例为一次性引擎调用，不启用服务会话缓存。调试 XXS 时同步替换模型路径、pack、量化文件名，并使用 `--spec 6`。

## 诊断顺序

1. 无 GPU 映射执行引擎 `--help`，检查产物、动态库和挂载路径。
2. 确认日志中的实际 GPU、权重类型、专家驻留率和 KV 模式。
3. 用短请求检查服务，再逐级增加输入长度。
4. 记录 prefill/TTFT、decode、MTP 接受率、显存峰值和系统内存；每次调整一项参数。
5. 比较性能时使用相同权重、输入、缓存状态和独占设备条件。
