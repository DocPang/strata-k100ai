# 从原版输入恢复源码与构建

**当前运行包使用冻结的历史 c4 二进制。下面的源码恢复已验证；完整干净编译、链接和新产物 GPU 等价验证尚未完成。** 历史构建曾复用旧目标文件，不能用旧增量构建脚本证明别人能够全量重建。

## 固定输入

- Strata：`https://github.com/Niko1221/Strata.git`，`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`。
- llama.cpp：`https://github.com/ggml-org/llama.cpp.git`，`3cf03257f219afbe7334045ff7c6a06ac68c627d`。
- DTK：使用 [INSTALL.md](INSTALL.md) 的固定镜像。
- 历史引擎 SHA-256：`c4be512a7d257d9b56bc325136431e5feffd2b6d76f52378051829bfe73fc903`。

## 恢复源码

在新工作目录执行，RECIPE 指向本仓库内配方绝对路径。开发者源码恢复工具需要 Python 3；普通安装用户不需要宿主 Python。

```bash
git clone https://github.com/Niko1221/Strata.git strata-original
git -C strata-original checkout --detach fb58e0dbc8399662c0e47c76578c6e878b14f6cf
git clone https://github.com/ggml-org/llama.cpp.git llama-original
git -C llama-original checkout --detach 3cf03257f219afbe7334045ff7c6a06ac68c627d
RECIPE=/absolute/path/strata-k100ai/engines/strata/c4/replay
python3 "$RECIPE/prepare.py" \
  --strata-source "$PWD/strata-original" \
  --llama-source "$PWD/llama-original" \
  --out "$PWD/prepared"
```

脚本检查原版每个文件、补丁和应用后的文件哈希；不会修改输入 checkout，也不覆盖已有输出目录。源码恢复成功覆盖 2,426 个 Strata 文件和 3,609 个 llama.cpp 文件。补丁为最终累计差异，不伪装成过去每轮实验的 Git 历史。

## 在固定 DTK 容器中构建

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
docker run --rm --network none --cpus=8 --memory=40g --pids-limit=2048 \
  -v "$PWD/prepared:/work" -v "$RECIPE:/recipe:ro" \
  --entrypoint bash "$BASE_IMAGE" /recipe/build-in-container.sh /work
```

此步骤不映射 GPU，不读取模型。它创建全新 build 目录，禁止复用旧 `.o`。只检查配置可加 `-e K100AI_CONFIGURE_ONLY=1`；不能把该模式的成功写成编译成功。

关键配方包括 `STRATA_HIP_GFX906=ON`、`STRATA_ENABLE_HIP=OFF`、目标 `gfx928`、gfx928 dot4、hipBLASLt、early-inline/no-function-calls。命名中的 gfx906 是该项目的 wave64 兼容后端，不代表目标硬件是 gfx906。脚本只在一次性构建容器中修正 HSA include 链接。

构建后需要重新做模型加载、数值/文本、固定负载 A/B/A、长上下文与缓存、服务接口和稳定性门槛。重新编译二进制 SHA 不同不自动判错；必须记录工具链和实际运行结果。不要覆盖正在使用的历史 Release 资产。
