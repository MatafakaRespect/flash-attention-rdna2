// fa_iface.h — ABI-контракт между torch-обёрткой (dispatch.cpp, host) и HIP-ядрами
// (fa_forward.hip forward-агента, backward.hip). Раскладки структур ДОЛЖНЫ побитово совпадать
// с определениями в .hip. Стрим передаётся как void* (hipStream_t — указатель, C-линковка
// сопоставляет по имени; типы игнорируются) — так dispatch.cpp не тянет hip/hip_runtime.h в MSVC.
#pragma once
#include <cstdint>

// --- forward (fa_forward.hip) ---
struct Strides { long long b, h, s; };
struct FwdParams {
    const void* Q; const void* K; const void* V; void* O; float* LSE;
    Strides sQ, sK, sV, sO;
    int SQ, SK, HD, NH, gqa;
    float scale, softcap;
    int win_left, win_right;   // семантика flash-attn: <0 => границы с этой стороны нет
    const float* alibi_slopes;
    int causal;
    float dropout_p; unsigned long long philox_seed;
    const int* cu_q; const int* cu_k;
    int batch;
    float sc_q, sc_k, sc_v;             // bf16 pow2-масштабы; 0 => ядро подставит 1
    int attn_chunk;                     // FA-3 attention_chunk; 0 => выключено
    const int* seqused_q; const int* seqused_k;   // FA-3 varlen; nullptr => вся длина
};

// --- backward (backward.hip) ---
struct Str { long long b, h, s; };
struct BwdParams {
    const void *Q,*K,*V,*O,*dO;  void *dQ,*dK,*dV;
    const float *lse; float *dsum;
    Str qs,ks,vs,os,dos,dqs,dks,dvs;
    int B,Hq,Hk,SQ,SK,D,gqa;
    float scale, softcap;
    const float *alibi;
    int causal, win_left, win_right;
    int dropout; unsigned long long drop_seed; float dropout_p, keep_scale;
    int shQ,shK,shV,shdO;
};

// --- decode с KV-кэшем (fa_decode.hip) ---
// Раскладка страничного кэша — канон flash-attn: [num_blocks, page, nheads_k, HD].
struct DecParams {
    const void* q;
    const void* Kc; const void* Vc;
    const int* block_table;
    void* o;
    float* scratch;              // [batch*NH*nsplit*(HD+2)]
    const int* ctx_len;
    const float* cos; const float* sin;
    int batch, NH, gqa, HD, page, max_blocks, nsplit, window;
    int page_shift, page_mask;   // выставляет сам launch
    int rotary;                  // 0 нет, 1 neox, 2 interleaved
    float scale;
    float sc_q, sc_k, sc_v;
    const int* leftpad;          // cache_leftpad[batch]; nullptr => 0
    float* lse_out;              // return_softmax_lse: [batch*nq, NH]; nullptr => не писать
    float softcap;               // <=0 => выключен
    const float* alibi;          // [NH] или nullptr
    int nq;                      // новых токенов за вызов (спекулятивка); 1 = обычный декод
    int append;                  // выставляет сам launch по new_k
    int rotary_dim;              // 0 => весь head_dim
    const int* rotary_pos;       // rotary_seqlens; nullptr => позиция берётся из ctx_len
};

extern "C" void fa_forward_launch(FwdParams p, int dtype, void* stream);
extern "C" void fable_attn_backward_launch(BwdParams pr, int MODE, void* stream);
extern "C" void fa_decode_launch(DecParams p, int dtype, void* stream,
                                 const void* new_k, const void* new_v);
