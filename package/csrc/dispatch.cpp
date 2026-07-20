// dispatch.cpp — torch-обёртка (host) над HIP-ядрами: forward-агента (fa_forward_launch) и
// backward (fable_attn_backward_launch). Регистрирует torch.ops.fable_attn.{fwd,bwd}.
// ROCm сохраняет CUDA dispatch-key для HIP-тензоров — это штатная карта torch, не хак.
#include <torch/extension.h>
#include <cmath>
#include "fa_iface.h"

// текущий torch-HIP-стрим берём из хипифаемой TU (stream.cu): в этом host .cpp (MSVC) нельзя
// тянуть hip-заголовки (ext_vector_type не парсится MSVC).
extern "C" void* fable_current_stream(long long dev);

// Лекарство линковки (PORT_NOTES_nvdiffrast.md): c10.dll не экспортирует унаследованный ctor
// c10::ValueError(SourceLocation,string) — подменяем на ctor базового Error. Без этой строки
// на этой машине не линкуется НИ ОДНО torch-расширение. Образец: D:\extensions\fable_attn\src\fable_attn.cpp.
#pragma comment(linker, "/alternatename:__imp_??0ValueError@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z=__imp_??0Error@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z")

static int dt_of(const torch::Tensor& t){
    if (t.scalar_type()==torch::kHalf)     return 0;
    if (t.scalar_type()==torch::kBFloat16) return 1;
    return 2; // float32
}
static int bf16_shift(const torch::Tensor& t, int dt){
    if (dt!=1) return 0;
    float m = t.abs().max().item<float>();
    if (m<=0.f) return 0;
    int e; std::frexp(m,&e);   // m in [2^(e-1),2^e)
    return 13 - e;
}
static void* cur_stream(const torch::Tensor& t){
    return fable_current_stream((long long)t.get_device());
}

// ============================ FORWARD ============================
// q:[B,Hq,SQ,D] k/v:[B,Hk,SK,D] (dim contiguous). Возвращает (out[B,Hq,SQ,D], lse[B,Hq,SQ] fp32).
static std::vector<torch::Tensor> fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    double scale, bool causal, int64_t win_left, int64_t win_right, double softcap,
    c10::optional<torch::Tensor> alibi, double dropout_p, int64_t seed,
    int64_t attn_chunk, c10::optional<torch::Tensor> out_opt)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "нужны GPU-тензоры");
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && v.dim()==4, "ожидается [B,H,S,D]");
    const int B=q.size(0), Hq=q.size(1), SQ=q.size(2), D=q.size(3);
    const int Hk=k.size(1), SK=k.size(2);
    TORCH_CHECK(Hq%Hk==0, "Hq должно делиться на Hk (GQA)");
    const int dt = dt_of(q), gqa = Hq/Hk;
    // out можно передать снаружи: обёртка выделяет его в раскладке (B,S,H,D), чтобы потребитель
    // получил ПЛОТНЫЙ тензор (как у настоящего flash-attn), а не транспонированный вид, на
    // котором падает .view(). Ядру всё равно: оно берёт страйды выхода отдельным полем.
    auto out = out_opt.has_value() ? *out_opt : torch::empty_like(q);
    if (out_opt.has_value()) {
        TORCH_CHECK(out.sizes()==q.sizes(), "out должен быть формы q [B,H,SQ,D]");
        TORCH_CHECK(out.scalar_type()==q.scalar_type(), "out должен быть того же dtype, что q");
        TORCH_CHECK(out.stride(3)==1, "out: последняя размерность обязана быть непрерывной");
    }
    auto lse = torch::empty({B,Hq,SQ}, q.options().dtype(torch::kFloat32));
    const float* al = alibi.has_value() ? alibi->data_ptr<float>() : nullptr;
    void* st = cur_stream(q);
    const size_t esz = q.element_size();
    auto S=[&](const torch::Tensor& x){ return Strides{ x.stride(0), x.stride(1), x.stride(2) }; };
    // forward пишет LSE[h*SQ+qq] без офсета батча => зовём по одному батчу (O тоже верен тривиально).
    for (int b=0; b<B; b++){
        FwdParams p{};
        p.Q=(const char*)q.data_ptr()+ (size_t)b*q.stride(0)*esz;
        p.K=(const char*)k.data_ptr()+ (size_t)b*k.stride(0)*esz;
        p.V=(const char*)v.data_ptr()+ (size_t)b*v.stride(0)*esz;
        p.O=(char*)out.data_ptr()   + (size_t)b*out.stride(0)*esz;
        p.LSE=lse.data_ptr<float>() + (size_t)b*Hq*SQ;
        p.sQ=S(q); p.sK=S(k); p.sV=S(v); p.sO=S(out);
        p.SQ=SQ; p.SK=SK; p.HD=D; p.NH=Hq; p.gqa=gqa;
        p.scale=(float)scale; p.softcap=(float)softcap;
        p.win_left=(int)win_left; p.win_right=(int)win_right;
        p.alibi_slopes=al; p.causal=causal?1:0;
        p.dropout_p=(float)dropout_p; p.philox_seed=(unsigned long long)seed;
        p.cu_q=nullptr; p.cu_k=nullptr; p.batch=1;
        p.attn_chunk=(int)attn_chunk; p.seqused_q=nullptr; p.seqused_k=nullptr;
        fa_forward_launch(p, dt, st);
    }
    return {out, lse};
}

// ============================ BACKWARD ============================
static std::vector<torch::Tensor> bwd(
    torch::Tensor dout, torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor lse,
    double scale, bool causal, int64_t win_left, int64_t win_right, double softcap,
    c10::optional<torch::Tensor> alibi, double dropout_p, int64_t seed)
{
    const int B=q.size(0), Hq=q.size(1), SQ=q.size(2), D=q.size(3);
    const int Hk=k.size(1), SK=k.size(2);
    const int dt = dt_of(q), gqa = Hq/Hk;
    auto dq = torch::empty_like(q), dk = torch::empty_like(k), dv = torch::empty_like(v);
    auto dsum = torch::empty({B,Hq,SQ}, q.options().dtype(torch::kFloat32));
    const float* al = alibi.has_value() ? alibi->data_ptr<float>() : nullptr;
    void* st = cur_stream(q);
    auto S=[&](const torch::Tensor& x){ return Str{ x.stride(0), x.stride(1), x.stride(2) }; };
    BwdParams pr{};
    pr.Q=q.data_ptr(); pr.K=k.data_ptr(); pr.V=v.data_ptr(); pr.O=out.data_ptr(); pr.dO=dout.data_ptr();
    pr.dQ=dq.data_ptr(); pr.dK=dk.data_ptr(); pr.dV=dv.data_ptr();
    pr.lse=lse.data_ptr<float>(); pr.dsum=dsum.data_ptr<float>();
    pr.qs=S(q); pr.ks=S(k); pr.vs=S(v); pr.os=S(out); pr.dos=S(dout);
    pr.dqs=S(dq); pr.dks=S(dk); pr.dvs=S(dv);
    pr.B=B; pr.Hq=Hq; pr.Hk=Hk; pr.SQ=SQ; pr.SK=SK; pr.D=D; pr.gqa=gqa;
    pr.scale=(float)scale; pr.softcap=(float)softcap; pr.alibi=al;
    pr.causal=causal?1:0; pr.win_left=(int)win_left; pr.win_right=(int)win_right;
    pr.dropout=(dropout_p>0)?1:0; pr.drop_seed=(unsigned long long)seed;
    pr.dropout_p=(float)dropout_p; pr.keep_scale=(dropout_p>0)?(float)(1.0/(1.0-dropout_p)):1.f;
    pr.shQ=bf16_shift(q,dt); pr.shK=bf16_shift(k,dt); pr.shV=bf16_shift(v,dt); pr.shdO=bf16_shift(dout,dt);
    fable_attn_backward_launch(pr, dt, st);
    return {dq, dk, dv};
}

// ============================ VARLEN FORWARD ============================
// Ядро умеет cu_seqlens нативно (FwdParams.cu_q / cu_k) — раньше обёртка это не использовала
// и гоняла питоновский цикл с отдельным запуском на каждую последовательность.
// q,k,v: (total_tokens, nheads, headdim). LSE не пишется (нужен только backward, а он плотный).
static torch::Tensor fwd_varlen(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor cu_q, torch::Tensor cu_k, int64_t max_sq, int64_t max_sk,
    double scale, bool causal, int64_t win_left, int64_t win_right, double softcap,
    c10::optional<torch::Tensor> alibi, double dropout_p, int64_t seed,
    int64_t attn_chunk,
    c10::optional<torch::Tensor> seqused_q, c10::optional<torch::Tensor> seqused_k)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "нужны GPU-тензоры");
    TORCH_CHECK(q.dim()==3 && k.dim()==3 && v.dim()==3, "varlen ждёт [total, H, D]");
    const int Hq=q.size(1), D=q.size(2), Hk=k.size(1);
    TORCH_CHECK(Hq%Hk==0, "Hq должно делиться на Hk (GQA)");
    const int dt = dt_of(q);
    const int batch = (int)cu_q.numel() - 1;

    // При seqused_q ядро не пишет строки за пределами используемой длины — из empty_like они
    // ушли бы наружу неинициализированным мусором (в замере это были NaN). Обнуляем ТОЛЬКО
    // в этом случае: на обычном varlen-пути лишнего memset нет.
    auto out = seqused_q.has_value() ? torch::zeros_like(q) : torch::empty_like(q);
    const float* al = alibi.has_value() ? alibi->data_ptr<float>() : nullptr;

    FwdParams p{};
    p.Q=q.data_ptr(); p.K=k.data_ptr(); p.V=v.data_ptr(); p.O=out.data_ptr(); p.LSE=nullptr;
    // (total,H,D): шаг по токену = H*D, по голове = D
    p.sQ=Strides{0, (long long)D, (long long)Hq*D};
    p.sK=Strides{0, (long long)D, (long long)Hk*D};
    p.sV=p.sK;
    p.sO=p.sQ;
    p.SQ=(int)max_sq; p.SK=(int)max_sk; p.HD=D; p.NH=Hq; p.gqa=Hq/Hk;
    p.scale=(float)scale; p.softcap=(float)softcap;
        p.win_left=(int)win_left; p.win_right=(int)win_right;
    p.alibi_slopes=al; p.causal=causal?1:0;
    p.dropout_p=(float)dropout_p; p.philox_seed=(unsigned long long)seed;
    p.cu_q=cu_q.data_ptr<int>(); p.cu_k=cu_k.data_ptr<int>();
    p.batch=batch;
    p.attn_chunk=(int)attn_chunk;
    p.seqused_q = seqused_q.has_value() ? seqused_q->data_ptr<int>() : nullptr;
    p.seqused_k = seqused_k.has_value() ? seqused_k->data_ptr<int>() : nullptr;
    fa_forward_launch(p, dt, cur_stream(q));
    return out;
}

// ============================ DECODE (KV-кэш) ============================
// Страничный кэш в раскладке flash-attn: k_cache/v_cache = [num_blocks, page, nheads_k, HD].
// block_table = [batch, max_blocks] (int32). cache_seqlens = [batch] (int32).
static std::vector<torch::Tensor> dec(
    torch::Tensor q,                       // [B, 1, Hq, D] — один новый токен
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor block_table, torch::Tensor cache_seqlens,
    c10::optional<torch::Tensor> new_k, c10::optional<torch::Tensor> new_v,
    c10::optional<torch::Tensor> rotary_cos, c10::optional<torch::Tensor> rotary_sin,
    bool rotary_interleaved, double scale, int64_t window, int64_t num_splits,
    c10::optional<torch::Tensor> cache_leftpad, bool return_lse,
    double softcap, c10::optional<torch::Tensor> alibi,
    int64_t rotary_dim, c10::optional<torch::Tensor> rotary_pos)
{
    TORCH_CHECK(q.is_cuda() && k_cache.is_cuda() && v_cache.is_cuda(), "нужны GPU-тензоры");
    TORCH_CHECK(q.dim()==4, "декод ждёт q формы [B,nq,Hq,D]");
    TORCH_CHECK(k_cache.dim()==4, "k_cache ждётся [num_blocks, page, nheads_k, D]");
    const int B=q.size(0), NQ=q.size(1), Hq=q.size(2), D=q.size(3);
    const int page=k_cache.size(1), Hk=k_cache.size(2);
    TORCH_CHECK(Hq%Hk==0, "Hq должно делиться на nheads_k (GQA)");
    TORCH_CHECK((page & (page-1))==0, "page должен быть степенью двойки");
    const int dt = dt_of(q);

    // nsplit: воркгруп обслуживает всю GQA-группу (K/V читаются раз на группу), значит
    // воркгрупов = batch*(Hq/gqa)*nsplit, и nsplit подбираем так, чтобы занять карту (~288).
    //
    // Верхнюю границу контекста берём ИЗ ФОРМ, а не из cache_seqlens.max().item():
    // .item() тянет значение с устройства на хост, то есть синхронизирует поток на каждом
    // вызове декода. При ядре в 0.26 мс это давало 0.7 мс накладных — втрое дороже самой работы.
    // Точное значение тут и не нужно: nsplit только режет контекст на срезы, а реальная длина
    // берётся в ядре из ctx_len для каждого элемента батча.
    const int gqa_r = Hq/Hk;
    const long ctx_bound = block_table.size(1) * (long)page;
    int nsplit;
    if (num_splits>0) nsplit = (int)num_splits;
    else {
        const int groups = B*NQ*(Hq/gqa_r);
        nsplit = (288 + groups - 1)/groups;
        int maxn = (int)((ctx_bound+255)/256); if (maxn<1) maxn=1;
        if (nsplit>maxn) nsplit=maxn;
    }
    nsplit = nsplit<1 ? 1 : (nsplit>512 ? 512 : nsplit);

    auto out = torch::empty({B,NQ,Hq,D}, q.options());
    auto scratch = torch::empty({(long)B*NQ*Hq*nsplit*(D+2)}, q.options().dtype(torch::kFloat32));
    // LSE-тензор выделяем только когда его попросили: пустой тензор -> lse_out=nullptr,
    // ядро сведения не делает лишней записи вообще.
    auto lse = return_lse ? torch::empty({B,NQ,Hq}, q.options().dtype(torch::kFloat32))
                          : torch::empty({0}, q.options().dtype(torch::kFloat32));

    DecParams p{};
    p.q=q.data_ptr(); p.Kc=k_cache.data_ptr(); p.Vc=v_cache.data_ptr();
    p.block_table=block_table.data_ptr<int>();
    p.o=out.data_ptr(); p.scratch=scratch.data_ptr<float>();
    p.ctx_len=cache_seqlens.data_ptr<int>();
    p.cos = rotary_cos.has_value() ? rotary_cos->data_ptr<float>() : nullptr;
    p.sin = rotary_sin.has_value() ? rotary_sin->data_ptr<float>() : nullptr;
    p.batch=B; p.nq=NQ; p.NH=Hq; p.gqa=Hq/Hk; p.HD=D; p.page=page;
    p.max_blocks=(int)block_table.size(1); p.nsplit=nsplit; p.window=(int)window;
    p.rotary = rotary_cos.has_value() ? (rotary_interleaved?2:1) : 0;
    p.scale=(float)scale; p.sc_q=1.f; p.sc_k=1.f; p.sc_v=1.f;
    p.leftpad = cache_leftpad.has_value() ? cache_leftpad->data_ptr<int>() : nullptr;
    p.lse_out = return_lse ? lse.data_ptr<float>() : nullptr;
    p.softcap=(float)softcap;
    p.alibi = alibi.has_value() ? alibi->data_ptr<float>() : nullptr;
    p.rotary_dim=(int)rotary_dim;
    p.rotary_pos = rotary_pos.has_value() ? rotary_pos->data_ptr<int>() : nullptr;

    const void* nk = new_k.has_value() ? new_k->data_ptr() : nullptr;
    const void* nv = new_v.has_value() ? new_v->data_ptr() : nullptr;
    fa_decode_launch(p, dt, cur_stream(q), nk, nv);
    return {out, lse};
}

// Регистрация через TORCH_LIBRARY здесь НЕ используется: её статические конструкторы
// выполняются во время LoadLibrary, и на Windows это роняло загрузку .pyd с
// ERROR_DLL_INIT_FAILED (1114) — расширение не грузилось вообще ни в одном окружении,
// а обёртка молча уходила в эталон. Соседнее рабочее расширение в этой же системе отдаёт
// функции через pybind, без регистрации в диспетчере на этапе загрузки DLL. Делаем так же:
// привязки создаются в PyInit, когда torch уже полностью поднят.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.doc() = "fable_attn HIP-ядра FlashAttention (fwd/bwd) для gfx1030";
    m.def("fwd", &fwd, "FlashAttention forward",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal"),
          py::arg("win_left"), py::arg("win_right"), py::arg("softcap"), py::arg("alibi"), py::arg("dropout_p"),
          py::arg("seed"), py::arg("attn_chunk")=0,
          py::arg("out")=c10::optional<torch::Tensor>());
    m.def("bwd", &bwd, "FlashAttention backward",
          py::arg("dout"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"),
          py::arg("lse"), py::arg("scale"), py::arg("causal"),
          py::arg("win_left"), py::arg("win_right"),
          py::arg("softcap"), py::arg("alibi"), py::arg("dropout_p"), py::arg("seed"));
    m.def("fwd_varlen", &fwd_varlen, "FlashAttention forward, varlen через cu_seqlens",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cu_q"), py::arg("cu_k"),
          py::arg("max_sq"), py::arg("max_sk"), py::arg("scale"), py::arg("causal"),
          py::arg("win_left"), py::arg("win_right"), py::arg("softcap"), py::arg("alibi"), py::arg("dropout_p"),
          py::arg("seed"), py::arg("attn_chunk")=0,
          py::arg("seqused_q")=c10::optional<torch::Tensor>(),
          py::arg("seqused_k")=c10::optional<torch::Tensor>());
    m.def("dec", &dec, "FlashAttention decode со страничным KV-кэшем",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("block_table"),
          py::arg("cache_seqlens"), py::arg("new_k"), py::arg("new_v"),
          py::arg("rotary_cos"), py::arg("rotary_sin"), py::arg("rotary_interleaved"),
          py::arg("scale"), py::arg("window"), py::arg("num_splits"),
          py::arg("cache_leftpad")=c10::optional<torch::Tensor>(),
          py::arg("return_lse")=false, py::arg("softcap")=0.0,
          py::arg("alibi")=c10::optional<torch::Tensor>(),
          py::arg("rotary_dim")=0,
          py::arg("rotary_pos")=c10::optional<torch::Tensor>());
}
