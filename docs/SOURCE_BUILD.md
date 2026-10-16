# 源码恢复与构建

rc.4 发布引擎（`5d87f09e07bb`）在 XXS exact+fused 基准引擎（`ad743dbf9cc1`）上增加 selection-batch 参数及配套显存记账，实际产物身份见 [新引擎记录](../engines/strata/xxs-exact-sel/build-record.json)。先按下述步骤恢复 c4 源码，再应用附加补丁：

```bash
git -C prepared/strata apply /absolute/path/strata-k100ai/engines/strata/xxs-exact-sel/patches/c4-to-exact-sel.patch
```

附加补丁仅修改 `src/prefill/prefill.cpp`，已验证输出与此次编译输入一致。实际发布二进制使用 XXS 基准引擎（`ad743dbf9cc1`）的静态对象和新 prefill 对象完成增量链接；下面的全量构建仍需另行完成和验收。

# c4 基础源码恢复

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

脚本校验原版输入、补丁与应用后文件的哈希，在新的输出目录中恢复源码，输入 checkout 保持只读。源码恢复成功覆盖 2,426 个 Strata 文件和 3,609 个 llama.cpp 文件。补丁为累计源码快照，逐次实验提交历史未纳入该快照。

## 在固定 DTK 容器中构建

```bash
BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/sglang@sha256:878d317dfa18efae70516a59cc836efc77b251efbe4ca85de67171bc2c3aca68
docker run --rm --network none --cpus=8 --memory=40g --pids-limit=2048 \
  -v "$PWD/prepared:/work" -v "$RECIPE:/recipe:ro" \
  --entrypoint bash "$BASE_IMAGE" /recipe/build-in-container.sh /work
```

构建使用 CPU 和源码输入，创建全新 build 目录。只检查配置可加 `-e K100AI_CONFIGURE_ONLY=1`；配置生成、编译链接与运行验收分别记录。

关键配方包括 `STRATA_HIP_GFX906=ON`、`STRATA_ENABLE_HIP=OFF`、目标 `gfx928`、gfx928 dot4、hipBLASLt、early-inline/no-function-calls。命名中的 gfx906 是该项目的 wave64 兼容后端，不代表目标硬件是 gfx906。脚本只在一次性构建容器中修正 HSA include 链接。

构建后需要重新做模型加载、数值/文本、固定负载 A/B/A、长上下文与缓存、服务接口和稳定性门槛。重新编译二进制 SHA 不同不自动判错；必须记录工具链和实际运行结果。不要覆盖正在使用的历史 Release 资产。
