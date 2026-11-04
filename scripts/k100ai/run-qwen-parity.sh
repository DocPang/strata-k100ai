#!/usr/bin/env bash
set -euo pipefail

root=/data/strata-dsv41-k100ai
image=${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10}
exe=${QWEN_EXE:-"$root/build-gfx928-qwen/strata"}
name=${CONTAINER_NAME:-strata-k100ai-qwen-parity-hcu0}
stamp=${LOG_STAMP:-"$(date +%Y%m%d-%H%M%S)"}
log=${LOG:-"$root/qwen-parity-hcu0-$stamp.log"}
logits=${LOGITS:-"$root/qwen-parity-hcu0-$stamp.logits"}
tokens=${TOKENS_FILE:-"$root/prompts/bench-zh.tokens"}
pack=${QWEN_PACK:-/data/experiments/strata_iq3xxs_gfx928_20261015/packs/iq3xxs}
native=${QWEN_NATIVE:-/data/experimental-models/qwen38-flash-next-gsq-rco-iq3-xxs-20261015/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf}
ple=${QWEN_PLE:-/data/experimental-models/qwen38-flash-next-gsq-rco-iq3-xxs-20261015/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf}
profile=${EXPERT_PROFILE:-/data/experiments/strata_iq3xxs_gfx928_20261015/strata/data/expert-profile.bin}
mtp=${QWEN_MTP_RT:-/data/experiments/strata_iq3xxs_gfx928_20261015/mtp-local-rt-r1}
export STRATA_K100AI_EXACT_MMAC=${STRATA_K100AI_EXACT_MMAC:-1}
export STRATA_K100AI_EXACT_GATE_FORMATS=${STRATA_K100AI_EXACT_GATE_FORMATS:-IQ3_XXS,IQ3_S,IQ4_XS}
export STRATA_K100AI_EXACT_DOWN_FORMATS=${STRATA_K100AI_EXACT_DOWN_FORMATS:-Q2_0}

research=(
  strata-next-hcu0-xxs-20261015
  strata-next-hcu1-xxs-20261016
  geak-k100ai-hcu23-20261030
)
running=()
extra_env=()
for key in HSA_FORCE_FINE_GRAIN_PCIE ROCPROF_EXE ROCPROF_OUT ROCPROF_DELAY_MS ROCPROF_ACTIVE_MS \
           STRATA_PREFILL_TIMING STRATA_TRACE STRATA_COLD_TRACE STRATA_GEMM_ATTR STRATA_HIPBLASLT_VERBOSE STRATA_ROCBLAS_VERBOSE STRATA_GDN_CHUNK_PROF STRATA_PREFILL_WARM STRATA_PREFILL_CPU_SHARE STRATA_VERIFY_PROFILE STRATA_ROUTE_STATS STRATA_MMVQ_PROF STRATA_WIN_TIMING STRATA_VERIFY_EARLY_PLE_FLAG \
           STRATA_IO_PREFETCH STRATA_IO_PF_GATE STRATA_COLD_OVERLAP \
           STRATA_DEQUANT_WIDE \
           STRATA_SPEC_T_CYCLE STRATA_DECODE_TIMING \
           STRATA_ATTN_LANECELL STRATA_ATTN_MERGE_V2 STRATA_HC_SPLIT STRATA_GR_DOWN_PREFETCH \
           STRATA_EXPERT_V2 STRATA_DOWN_MT STRATA_Q6_WAVE_PACK STRATA_Q6_WAVE_PACK_PREP \
           STRATA_Q6_PACKED STRATA_Q8_PACKED STRATA_GDN_HEAD STRATA_GDN_NOY \
           STRATA_GDN_PP STRATA_GDN_CONVL2 STRATA_GDN_SPLIT STRATA_TSUM STRATA_LFUSE \
           STRATA_QFUSE STRATA_PLE_BATCH STRATA_SH_STREAM STRATA_MMVF_ROWS STRATA_EXPDEC \
           STRATA_HIP_WMMA STRATA_HIP_PROMPT_F16 STRATA_MMQ_EXACT_ALL STRATA_EXP_MODE \
           STRATA_COMB_VEC STRATA_HC_UPMIX STRATA_HC_UPMIX_CHECK \
           STRATA_PF_FUSED STRATA_PF_FUSED_NATIVE_EXACT_SKIP STRATA_PF_FUSED_SQ \
           STRATA_PF_PAD_ROCBLAS STRATA_PF_HCDOWN STRATA_HCD_EXACT STRATA_KV_PREFETCH STRATA_TOPK_H16 \
           STRATA_TOPK_REG_MAX STRATA_SELECT_LONG STRATA_SCORES_THREAD \
           STRATA_MMVQ_WV2_LDS STRATA_MMVQ_WV2_GROUP STRATA_Q6_WAVE_PACK_LDS \
           STRATA_EMB_REUSE_ACCOUNT STRATA_GATEUP_EXACT_IQ2 STRATA_GEMM_Y_F16 \
           STRATA_HC_UPMIX_GROUPS STRATA_MMQ_GATEUP_MINROWS \
           STRATA_MMQ_RESIDENT_SORT_NATIVE STRATA_MMQ_RESIDENT_SORT_NE \
           STRATA_PF_DMSCRATCH_COND STRATA_PF_GATED_IN_REGION STRATA_PLE_PRELAUNCH \
           STRATA_PREFILL_RING STRATA_SCORES_MMAC STRATA_SCORES_MMAC_LDS \
           STRATA_SCORES_MMAC_NB STRATA_SEL_SCORES_FP16 STRATA_K100AI_HC_PARALLEL \
           STRATA_K100AI_EXACT_MMAC STRATA_K100AI_EXACT_GATEUP \
           STRATA_K100AI_EXACT_DOWN STRATA_K100AI_EXACT_GATE_FORMATS \
           STRATA_K100AI_EXACT_DOWN_FORMATS; do
  if value=$(printenv "$key"); then
    extra_env+=(-e "$key=$value")
  fi
done

cleanup() {
  docker rm -f "$name" >/dev/null 2>&1 || true
  for container in "${running[@]}"; do
    if ! docker start "$container" >/dev/null 2>&1; then
      echo "failed to restore research container: $container" >&2
    fi
  done
}
trap cleanup EXIT

if [[ ! -x "$exe" ]]; then
  echo "missing executable: $exe" >&2
  exit 2
fi
for mtp_file in experts.bin dense.bin dense.txt draft_vocab.bin; do
  if [[ ! -f "$mtp/$mtp_file" ]]; then
    echo "missing Qwen MTP runtime file: $mtp/$mtp_file" >&2
    exit 2
  fi
done

for container in "${research[@]}"; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" == true ]]; then
    running+=("$container")
  fi
done

for container in "${running[@]}"; do
  docker stop --time 30 "$container" >/dev/null
done

bash "$root/scripts/k100ai/preflight.sh"

libs=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib:
set +e
docker run --rm --name "$name" \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --shm-size=8g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data:/data \
  -w "$root" \
  -e LD_LIBRARY_PATH="$libs" \
  -e HIP_VISIBLE_DEVICES=0 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e STRATA_DUMP_FIRST_LOGITS="$logits" \
  "${extra_env[@]}" \
  --entrypoint "$exe" \
  "$image" \
  --pack "$pack" \
  --native "$native" \
  --ple-gguf "$ple" \
  --ple-io mmap \
  --mmap-experts \
  --expert-cache "${QWEN_EXPERT_CACHE:-auto}" \
  --expert-profile "$profile" \
  --prefill "${QWEN_PREFILL_CHUNK:-20480}" \
  --spec "${QWEN_SPEC:-4}" \
  --mtp "$mtp" \
  --suffix-draft "${QWEN_SUFFIX_DRAFT:-0}" \
  --max-context "${QWEN_MAX_CONTEXT:-4096}" \
  --tokens-file "$tokens" \
  --max-new "${QWEN_MAX_NEW:-16}" \
  --dump-logits "$logits" \
  "$@" 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e

exit "$rc"
