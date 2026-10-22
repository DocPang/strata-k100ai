# 运行与调试

完整的下载、转换和启动步骤见[首页](../README.md#安装与运行)。本节的 `/srv` 路径是独立示例，使用前应按实际目录替换。

推荐实例使用 IQ3_XXS、全 KV 显存驻留、全部专家常驻和单槽 FIFO。先按 [INSTALL.md](INSTALL.md) 取得 DTK 镜像、权重和运行文件。

## 原生服务

在项目根目录执行，替换模型、pack 与 MTP 路径，并确认所选 render 节点对应空闲显卡：

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
RENDER_DEVICE=/dev/dri/renderD128
export STRATA_API_KEY="$(openssl rand -hex 24)"

docker run --rm -it --name strata-iq3xxs \
  --device /dev/kfd:/dev/kfd --device "$RENDER_DEVICE:$RENDER_DEVICE" \
  --ulimit memlock=-1:-1 --shm-size=2g \
  --security-opt seccomp=unconfined \
  --tmpfs /var/log/hylog:rw,nosuid,nodev,mode=0755 \
  --publish 127.0.0.1:18640:8080 --env STRATA_API_KEY \
  --env-file "$PWD/engines/strata/xxs-20261022-sol/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/models/next-iq3xxs/IQ3_XXS,dst=/models,readonly \
  --mount type=bind,src=/srv/packs/next-iq3xxs,dst=/pack,readonly \
  --mount type=bind,src=/srv/mtp-work/runtime,dst=/mtp,readonly \
  --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
  --mount "type=bind,src=$PWD/profiles/next-iq3xxs-single/strata.json,dst=/config/strata.json,readonly" \
  --workdir /opt/k100ai/app --entrypoint python3 \
  "$BASE_IMAGE" -m serve.server --engine strata \
  --config /config/strata.json --host 0.0.0.0 --port 8080
```

只开放一个 render 节点，并保留其原路径；容器内使用 GPU 0。`environment.env` 是优化环境，`strata.json` 是原生服务配置，可直接编辑 `args`，重建容器后生效。

## 调试与生命周期

`--rm -it` 为前台运行；替换成 `--detach` 可在后台运行并保留容器。原生实例的服务日志和引擎日志分别查看：

```bash
docker logs -f strata-iq3xxs
docker exec -it strata-iq3xxs tail -f /tmp/strata-engine.log
curl --fail http://127.0.0.1:18640/health
curl --fail -H "Authorization: Bearer $STRATA_API_KEY" http://127.0.0.1:18640/v1/models
```

上述命令分别执行，`logs -f` 与 `tail -f` 会持续跟踪。API 请求示例：

```bash
curl --fail http://127.0.0.1:18640/v1/chat/completions \
  -H "Authorization: Bearer $STRATA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-strata","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"stream":true}'
```

API key 需与对应实例一致。后台容器停止后手动删除；前台 `--rm` 模式退出后会自动删除：

```bash
docker stop strata-iq3xxs
docker rm strata-iq3xxs
```

## 参数调整

默认不传 `--kv-resident`、`--batch`、`--batch-mtp`。如需恢复 KV streaming 对照，可显式添加 `--kv-resident 32768`；它是容量回退选项，不是 64 GiB XXS 的默认配置。

本版不设置 `STRATA_SEL_BATCH` 覆盖值，使用引擎默认分档。它控制 selection 的 query 批次，不增加会话并发。关键优化变量及作用见 [OPTIMIZATIONS.md](OPTIMIZATIONS.md)。

prefill 工作区受限时将 `--prefill 30602` 先降至 20480，再降至 16384。最大上下文为 262144；本版不提供 1M 外推配置。默认关闭 prompt/conversation cache，开启缓存后需重新检查主机内存和延迟。容量和成绩条件见 [MEMORY.md](MEMORY.md) 与 [BENCHMARKS.md](BENCHMARKS.md)。

## 多卡并发

每张卡运行独立的模型实例和 FIFO 队列，使用不同容器名、render 节点和宿主端口，例如 18640、18641。网关将请求分发到不同实例。这是数据并行，不是 TP2。新增实例前检查设备占用，聚合性能按实际请求和共享 CPU/内存条件测量。

## 直接调用 C++ 引擎

需要绕过 HTTP 前端分析加载、prefill 或 decode 时，可直接运行引擎。输入为与当前 tokenizer 匹配的 token ID 文件，不是 UTF-8 提示词。先将预分词结果保存在本机 `prompt.ids`。

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
RENDER_DEVICE=/dev/dri/renderD128

docker run --rm -it --name strata-iq3xxs-engine \
  --device /dev/kfd:/dev/kfd \
  --device "$RENDER_DEVICE:$RENDER_DEVICE" \
  --ulimit memlock=-1:-1 --shm-size=2g \
  --security-opt seccomp=unconfined \
  --tmpfs /var/log/hylog:rw,nosuid,nodev,mode=0755 \
  --env-file "$PWD/engines/strata/xxs-20261022-sol/environment.env" \
  --mount "type=bind,src=$PWD/runtime,dst=/opt/k100ai,readonly" \
  --mount type=bind,src=/srv/models/next-iq3xxs/IQ3_XXS,dst=/models,readonly \
  --mount type=bind,src=/srv/packs/next-iq3xxs,dst=/pack,readonly \
  --mount type=bind,src=/srv/mtp-work/runtime,dst=/mtp,readonly \
  --mount type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly \
  --mount "type=bind,src=$PWD/prompt.ids,dst=/input/prompt.ids,readonly" \
  --workdir /opt/k100ai/app --entrypoint /opt/k100ai/bin/strata \
  "$BASE_IMAGE" \
  --gpu 0 --pack /pack \
  --native /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --ple-io mmap --mmap-experts \
  --expert-cache 24576 --expert-profile /opt/k100ai/app/data/expert-profile.bin \
  --vram-reserve-mib 2048 --prefill 30602 \
  --max-context 262144 --kv int8 \
  --mtp /mtp --spec 4 --spec-min-p 0.70 --pool-workers 12 \
  --eos-ids 248046 --no-prefill-borrow \
  --tokens-file /input/prompt.ids --max-new 128
```

输入 token ID 必须由对应 tokenizer 生成。该示例为一次性引擎调用，不启用服务会话缓存。

## 历史配置

IQ3_S/c4 模板保留用于复核旧成绩，当前部署建议使用 XXS。辅助 `install.sh` 默认选择 XXS；显式选择 `--profile next-iq3s-single` 时使用旧 c4 运行包，不能把新旧引擎校验值混用。
