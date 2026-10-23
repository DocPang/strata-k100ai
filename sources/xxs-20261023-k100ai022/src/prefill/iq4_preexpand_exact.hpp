#include "kdn_down_gemm.hpp"

// Exact code expansion; packed IQ4_NL remains canonical. D4 Q8 layout is unchanged.
struct Q8D4 { float d[4]; int8_t q[128]; };
static_assert(sizeof(Q8D4)==144,"Q8 D4 layout");
using I4 = int __attribute__((vector_size(16)));
// downstage-20261013: the ORIGINAL table, kept only for the same-binary A/B (STRATA_DOWN_TBL=0).
__device__ __constant__ int8_t iq4_table_cst[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
// Plain __device__ (global), NOT __constant__: every lane looks up a DIFFERENT idx, and divergent
// __constant__ reads serialize (the same reason gateup_exact.hpp keeps G3D/K3D/G2D/GSD/KND in global
// memory).  The 16 bytes are one cache line, so the per-lane global reads coalesce.  STRATA_DOWN_TBL=0
// restores the __constant__ table.
__device__ int8_t iq4_table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
template<bool CTBL>
__global__ void expand_iq4_k(const uint8_t* w,int8_t* codes,float* scales,int blocks,const int* bounds,int ne){
 int e=blockIdx.z;if(bounds[e]==bounds[e+1])return;int bpe=blocks/ne;int localword=blockIdx.x*blockDim.x+threadIdx.x;if(localword>=bpe*4)return;int word=e*bpe*4+localword;
 int b=word/4,seg=word%4;const uint8_t* p=w+(size_t)b*18;
 if(seg==0)scales[b]=__half2float(*reinterpret_cast<const __half*>(p));
 uint64_t v=0;
 #pragma unroll
 for(int i=0;i<8;i++){unsigned q=p[2+(seg%2)*8+i];unsigned idx=(q>>(seg<2?0:4))&15;v|=(uint64_t)(uint8_t)(CTBL?iq4_table_cst[idx]:iq4_table[idx])<<(i*8);}
 reinterpret_cast<uint64_t*>(codes)[word]=v;
}
// Host dispatcher: STRATA_DOWN_TBL=0 -> the original __constant__ table, anything else -> global.
void expand_iq4_launch(const uint8_t* w,int8_t* codes,float* scales,int blocks,const int* bounds,int ne,hipStream_t s){
 static const bool cst=[](){const char* v=std::getenv("STRATA_DOWN_TBL");return v!=nullptr&&std::atoi(v)==0;}();
 const dim3 g((unsigned)(((blocks/ne)*4+255)/256),1,(unsigned)ne);
 if(cst)expand_iq4_k<true ><<<g,256,0,s>>>(w,codes,scales,blocks,bounds,ne);
 else   expand_iq4_k<false><<<g,256,0,s>>>(w,codes,scales,blocks,bounds,ne);
}
// downstage-20261013: the Q2_0 (d_type 42) down expander.  A GGUF q2_0 block is {f16 d; 16 code bytes}
// per 64 values with value = d*(q-1), q = 2 bits, four codes per byte
// (ggml-quants.c dequantize_row_q2_0: 00=-1, 01=0, 10=+1, 11=+2).  So codes[k] = q-1 in {-1,0,1,2} and
// the block's single d goes into BOTH of its K/32 scale slots: the scaled int8 dot the GEMM computes is
// then exactly what the dp4a MMQ computes.  A q2_0 block is 64 values, so this needs 8 words of 8 bytes
// per block (expand_iq4 needs 4: its block is 32 values), and no table
// lookup at all (q-1 is arithmetic, so there is no divergent constant/global read).
__global__ void expand_q2_0_k(const uint8_t* w,int8_t* codes,float* scales,int blocks,const int* bounds,int ne){
 int e=blockIdx.z;if(bounds[e]==bounds[e+1])return;int bpe=blocks/ne;int localword=blockIdx.x*blockDim.x+threadIdx.x;if(localword>=bpe*8)return;int word=e*bpe*8+localword;
 int b=word/8,seg=word%8;const uint8_t* p=w+(size_t)b*18;
 if(seg==0){const float d=__half2float(*reinterpret_cast<const __half*>(p));scales[b*2]=d;scales[b*2+1]=d;}
 uint64_t v=0;
 #pragma unroll
 for(int i=0;i<8;i++){unsigned q=p[2+2*seg+(i>>2)];unsigned idx=(q>>((i&3)*2))&3;v|=(uint64_t)(uint8_t)((int)idx-1)<<(i*8);}
 reinterpret_cast<uint64_t*>(codes)[word]=v;
}
void expand_q2_0_launch(const uint8_t* w,int8_t* codes,float* scales,int blocks,const int* bounds,int ne,hipStream_t s){
 const dim3 g((unsigned)(((blocks/ne)*8+255)/256),1,(unsigned)ne);
 expand_q2_0_k<<<g,256,0,s>>>(w,codes,scales,blocks,bounds,ne);
}
// f16-20261014 ("f16 Dm"): the down product's destination Dm is FP16 now, so this A/B kernel (STRATA_DOWN_STAGED=0)
// stores through gateup_exact::out_hf() like the staged GEMM's OT=uint16_t instantiation (same rounding, same order).
__global__ void expanded_iq4_gemm(const int8_t* codes,const float* scales,const Q8D4* x,const int* bounds,const int* dst,uint16_t* y,int N,int K,int rows){
 const int lane=threadIdx.x&63,wave=threadIdx.x/64,e=blockIdx.z;
 const int nr=blockIdx.x*64+wave*16+lane%16;
 const int cb=blockIdx.y*32,start=bounds[e],end=bounds[e+1];if(start+cb>=end)return;
 float acc[2][4]={{0,0,0,0},{0,0,0,0}};
 for(int k=0;k<K/32;k++){
  long av=nr<N?*reinterpret_cast<const long*>(codes+((size_t)e*N+nr)*K+k*32+(lane/16)*8):0;
  const float ws=nr<N?scales[((size_t)e*N+nr)*(K/32)+k]:0.f;
  #pragma unroll
  for(int t=0;t<2;t++){
   const int br=start+cb+t*16+lane%16;
   long bv=br<end?*reinterpret_cast<const long*>(x[(k/4)*rows+br].q+(k%4)*32+(lane/16)*8):0;
   I4 c={0,0,0,0};c=__builtin_amdgcn_mmac_i32_16x16x32i8(av,bv,c);
   #pragma unroll
   for(int j=0;j<4;j++){int col=start+cb+t*16+lane/16+4*j;float xs=col<end?x[(k/4)*rows+col].d[k%4]:0.f;acc[t][j]=fmaf(float(c[j]),ws*xs,acc[t][j]);}
  }
 }
 #pragma unroll
 for(int t=0;t<2;t++){
  #pragma unroll
  for(int j=0;j<4;j++){int col=start+cb+t*16+lane/16+4*j;if(nr<N&&col<end)y[(size_t)dst[col]*N+nr]=gateup_exact::out_hf(acc[t][j]);}
 }
}
// downstage-20261013: the down product's GEMM now runs gateup_exact::gemm_moe_staged (the LDS-staged
// grouped MMAC of gateup_exact.hpp) instead of expanded_iq4_gemm, which re-read every activation row
// w_rows/64 times and every weight row max_rows/32 times straight from HBM and spent 8 of its 12 global
// loads per k window on the activation scale.  See REPORT_downstage.md.
// Layouts are unchanged (int8 codes [N][K] row-major, fp32 scales [N][K/32], Q8D4 activations
// [K/128][rows][128], bounds/ids/dst, output dst[activation_row]*N + weight_row) and so is the per-output
// accumulation order (one 16x16x32 MMAC + one fmaf per 32-wide k window, k ascending), so the result is
// bit-identical to expanded_iq4_gemm -- verified on real histograms and end to end by output_sha256.
// STRATA_DOWN_STAGED=0 restores the old kernel; STRATA_DOWN_TILE selects the N tile (default 8).
// The GEMM is shared with the Q2_0 down path (run_expanded_q2_0 below): both feed it an int8 [N][K]
// code plane and a fp32 [N][K/32] scale plane, which is all it reads.
// downmt-20261014: rows-per-CTA knob for the down product.
// The down GEMM has K=640 (20 k-windows) against gate/up's K=2560 (80), so every CTA amortises its
// prologue, LDS staging and epilogue over 4x fewer MACs -- the component scans put down at ~9.0
// TMAC/s against ~17.3 for gate/up with the *same* kernel. MT only changes which rows a CTA owns
// (BM = 64*MT); the per-output reduction order (k ascending, one MMAC per 32-wide window) is
// untouched, so the result stays bit-identical.
template<int MT,int NT,int KS>
static void down_gemm_launch_t(const Product &p,const int8_t* codes,const float* scales,hipStream_t s){
 constexpr int BM=16*MT*4,BN=16*NT;
 dim3 g((unsigned)((p.max_rows+BM-1)/BM),(unsigned)((p.w_rows+BN-1)/BN),(unsigned)p.n);
 // f16-20261014 ("f16 Dm"): the down product's destination is FP16 - OT=uint16_t makes the epilogue round once and
 // write half the bytes (the same store, the same k order and the same FP32 accumulation as before).
 gateup_exact::gemm_moe_staged<MT,NT,KS,32,true,uint16_t><<<g,256,0,s>>>((const int8_t*)codes,scales,(const gateup_exact::Q8D4*)p.xq,p.bounds,p.ids,p.dst16,p.w_rows,p.w_cols,(int)p.total_rows);
}
// kdn-20261014: the down GEMM is routed to the kdn_down kernel family (kdn_down_gemm.hpp), which is the
// bit-identical restructure described there.  Everything the launcher used to mean is preserved:
//   * the MT rule is unchanged (STRATA_DOWN_MT forces; otherwise max_rows >= STRATA_DOWN_MT_MINR (600) ? 2 : 1)
//   * STRATA_DOWN_TILE and STRATA_DOWN_STAGED keep their semantics; tile != 8 and mode == 0 (the old
//     expanded_iq4_gemm) still use the shipped kernels exactly as before, as does mt == 4
//   * STRATA_DOWN_KERNEL=0 restores the shipped dispatch for every setting (same-binary A/B switch)
// See kdn-20261014/REPORT.md.
static void down_gemm_launch(const Product &p,const int8_t* codes,const float* scales,hipStream_t s){
 static const int mode=[](){const char* v=std::getenv("STRATA_DOWN_STAGED");return v?std::atoi(v):1;}();
 static const int tile=[](){const char* v=std::getenv("STRATA_DOWN_TILE");return v?std::atoi(v):8;}();
 // downmt3-20261015: MT is SHAPE-ADAPTIVE. MT=2 (BM=128) wins when a group's longest expert is
 // large (32K: max_rows 793-1109+, measured -283..-300 ms) but LOSES when groups are small
 // (8K chunk: max_rows only 365-395, measured 1437 -> 1628 ms). Same pattern as the proven
 // wsg16 tile crossover in moe_mmq.cu. Default threshold 600; STRATA_DOWN_MT=<n> forces a value.
 static const int mt_fixed=[](){const char* v=std::getenv("STRATA_DOWN_MT");return v?std::atoi(v):0;}();
 static const int mt_thresh=[](){const char* v=std::getenv("STRATA_DOWN_MT_MINR");return v?std::atoi(v):600;}();
 const int mt = mt_fixed ? mt_fixed : (p.max_rows >= mt_thresh ? 2 : 1);
 static const int kdn_kernel=[](){const char* v=std::getenv("STRATA_DOWN_KERNEL");return v?std::atoi(v):1;}();
 if(kdn_kernel && mode!=0 && tile==8 && (mt==1 || mt==2)){
  if(mt==2) kdn_down::launch<2,8,64>(codes,scales,(const kdn_down::Q8D4*)p.xq,p.bounds,p.dst16,(int)p.w_rows,(int)p.w_cols,(int)p.total_rows,(int)p.max_rows,p.n,s);
  else      kdn_down::launch<1,8,64>(codes,scales,(const kdn_down::Q8D4*)p.xq,p.bounds,p.dst16,(int)p.w_rows,(int)p.w_cols,(int)p.total_rows,(int)p.max_rows,p.n,s);
  return;
 }
 if(mode==0){
  expanded_iq4_gemm<<<dim3((p.w_rows+63)/64,(p.max_rows+31)/32,p.n),256,0,s>>>(codes,scales,(const Q8D4*)p.xq,p.bounds,p.ids,p.dst16,p.w_rows,p.w_cols,p.total_rows);
 }else if(mt==4&&tile!=16) down_gemm_launch_t<4,8,64>(p,codes,scales,s);
 else if(mt==2&&tile==4)    down_gemm_launch_t<2,4,64>(p,codes,scales,s);
 else if(mt==2&&tile==16)   down_gemm_launch_t<2,16,64>(p,codes,scales,s);
 else if(mt==2)             down_gemm_launch_t<2,8,64>(p,codes,scales,s);
 else if(tile==4)           down_gemm_launch_t<1,4,64>(p,codes,scales,s);
 else if(tile==16)          down_gemm_launch_t<1,16,64>(p,codes,scales,s);
 else                       down_gemm_launch_t<1,8,64>(p,codes,scales,s);
}

void run_expanded_iq4(const Product &p,int8_t* codes,float* scales,hipStream_t s,bool convert=true){
 int blocks=p.n*p.w_rows*(p.w_cols/32);
 if(convert)expand_iq4_launch((const uint8_t*)p.w,codes,scales,blocks,p.bounds,p.n,s);
 // Phase boundary for STRATA_PREFILL_TIMING (prefill.cpp installs it; null otherwise, so no cost).
 if(strata::prefill::mmq::iq4_down_split_mark)strata::prefill::mmq::iq4_down_split_mark((void*)s);
 down_gemm_launch(p,codes,scales,s);
}
// Q2_0 (d_type 42) down: the same GEMM fed by the q2_0 expander.  Opt-in/out with
// STRATA_MMQ_Q2_0_PREEXPAND (default on); off means the production dp4a mul_mat_q_case<Q2_0>.
// k100ai/gfx928: the same launch, writing FP32 into the caller's buffer (no widen pass).
static void down_gemm_launch_f32(const Product &p,const int8_t* codes,const float* scales,hipStream_t s){
 static const int mt_fixed=[](){const char* v=std::getenv("STRATA_DOWN_MT");return v?std::atoi(v):0;}();
 static const int mt_thresh=[](){const char* v=std::getenv("STRATA_DOWN_MT_MINR");return v?std::atoi(v):600;}();
 const int mt = mt_fixed ? mt_fixed : (p.max_rows >= mt_thresh ? 2 : 1);
 if(mt==2) kdn_down::launch<2,8,64,0,true>(codes,scales,(const kdn_down::Q8D4*)p.xq,p.bounds,p.dst,(int)p.w_rows,(int)p.w_cols,(int)p.total_rows,(int)p.max_rows,p.n,s);
 else      kdn_down::launch<1,8,64,0,true>(codes,scales,(const kdn_down::Q8D4*)p.xq,p.bounds,p.dst,(int)p.w_rows,(int)p.w_cols,(int)p.total_rows,(int)p.max_rows,p.n,s);
}
// k100ai/gfx928: the IQ4_NL down product through the same expander, writing FP32 into the caller's buffer
// (our Dm is FP32; rc.7's down_gemm_launch targets the FP16 dst16).
void run_expanded_iq4_f32(const Product &p,int8_t* codes,float* scales,hipStream_t s,bool convert=true){
 int blocks=p.n*p.w_rows*(p.w_cols/32);
 if(convert)expand_iq4_launch((const uint8_t*)p.w,codes,scales,blocks,p.bounds,p.n,s);
 if(strata::prefill::mmq::iq4_down_split_mark)strata::prefill::mmq::iq4_down_split_mark((void*)s);
 down_gemm_launch_f32(p,codes,scales,s);
}
void run_expanded_q2_0_f32(const Product &p,int8_t* codes,float* scales,hipStream_t s,bool convert=true){
 int blocks=p.n*p.w_rows*(p.w_cols/64);
 if(convert)expand_q2_0_launch((const uint8_t*)p.w,codes,scales,blocks,p.bounds,p.n,s);
 if(strata::prefill::mmq::iq4_down_split_mark)strata::prefill::mmq::iq4_down_split_mark((void*)s);
 down_gemm_launch_f32(p,codes,scales,s);
}
void run_expanded_q2_0(const Product &p,int8_t* codes,float* scales,hipStream_t s,bool convert=true){
 int blocks=p.n*p.w_rows*(p.w_cols/64);
 if(convert)expand_q2_0_launch((const uint8_t*)p.w,codes,scales,blocks,p.bounds,p.n,s);
 if(strata::prefill::mmq::iq4_down_split_mark)strata::prefill::mmq::iq4_down_split_mark((void*)s);
 down_gemm_launch(p,codes,scales,s);
}
