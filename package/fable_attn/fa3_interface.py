"""Порядок аргументов FA-3 (пакет `flash_attn_interface` из hopper/).

Зачем отдельный модуль. У FA-2 и FA-3 РАЗНЫЙ порядок позиционных аргументов:

    FA-2: flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, ...)
    FA-3: flash_attn_func(q, k, v, softmax_scale, causal, qv, ...)

То есть четвёртый позиционный у FA-2 — вероятность dropout, а у FA-3 — масштаб softmax.
Одной функцией это не покрыть: вызов `flash_attn_func(q,k,v,0.088)` в одном соглашении значит
«dropout 8.8%», в другом — «scale 0.088», и угадывать по значению нельзя — так молча считают
не то. Поэтому:

    import flash_attn                 -> порядок FA-2 (fable_attn.flash_attn_interface)
    import flash_attn_interface       -> порядок FA-3 (этот модуль)

Оба зовут одни и те же ядра. Разница только в раскладке аргументов и в том, что FA-3 при
return_attn_probs отдаёт пару (out, softmax_lse), а FA-2 — тройку с матрицей вероятностей.
Регистрацию делает fable_attn.install_as_flash_attn().
"""
from . import flash_attn_interface as _fa2

__all__ = ["flash_attn_func", "flash_attn_qkvpacked_func", "flash_attn_varlen_func",
           "flash_attn_with_kvcache", "flash_attn_combine", "get_scheduler_metadata"]


def flash_attn_func(q, k, v, softmax_scale=None, causal=False, qv=None,
                    q_descale=None, k_descale=None, v_descale=None,
                    window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                    num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
                    return_attn_probs=False):
    r = _fa2.flash_attn_func(q, k, v, 0.0, softmax_scale, causal, window_size, softcap,
                             None, deterministic, return_attn_probs,
                             qv=qv, attention_chunk=attention_chunk,
                             q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
                             num_splits=num_splits, pack_gqa=pack_gqa, sm_margin=sm_margin)
    return (r[0], r[1]) if return_attn_probs else r   # FA-3 отдаёт пару, без матрицы S×S


def flash_attn_qkvpacked_func(qkv, softmax_scale=None, causal=False,
                              q_descale=None, k_descale=None, v_descale=None,
                              window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                              num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
                              return_attn_probs=False):
    r = _fa2.flash_attn_qkvpacked_func(qkv, 0.0, softmax_scale, causal, window_size, softcap,
                                       None, deterministic, return_attn_probs,
                                       attention_chunk=attention_chunk,
                                       q_descale=q_descale, k_descale=k_descale,
                                       v_descale=v_descale, num_splits=num_splits,
                                       pack_gqa=pack_gqa, sm_margin=sm_margin)
    return (r[0], r[1]) if return_attn_probs else r


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           seqused_q=None, seqused_k=None, softmax_scale=None, causal=False,
                           qv=None, q_descale=None, k_descale=None, v_descale=None,
                           window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                           num_splits=1, pack_gqa=None, deterministic=False, sm_margin=0,
                           return_attn_probs=False):
    r = _fa2.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                                    max_seqlen_k, 0.0, softmax_scale, causal, window_size,
                                    softcap, None, deterministic, return_attn_probs,
                                    seqused_q=seqused_q, seqused_k=seqused_k, qv=qv,
                                    attention_chunk=attention_chunk,
                                    q_descale=q_descale, k_descale=k_descale,
                                    v_descale=v_descale, num_splits=num_splits,
                                    pack_gqa=pack_gqa, sm_margin=sm_margin)
    return (r[0], r[1]) if return_attn_probs else r


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, qv=None,
                            rotary_cos=None, rotary_sin=None, cache_seqlens=None,
                            cache_batch_idx=None, cache_leftpad=None, page_table=None,
                            cu_seqlens_q=None, cu_seqlens_k_new=None, max_seqlen_q=None,
                            rotary_seqlens=None, q_descale=None, k_descale=None,
                            v_descale=None, softmax_scale=None, causal=False,
                            window_size=(-1, -1), attention_chunk=0, softcap=0.0,
                            rotary_interleaved=True, scheduler_metadata=None,
                            num_splits=0, pack_gqa=None, sm_margin=0,
                            return_softmax_lse=False):
    return _fa2.flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, qv=qv, rotary_cos=rotary_cos, rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens, cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad, block_table=None, softmax_scale=softmax_scale,
        causal=causal, window_size=window_size, softcap=softcap,
        rotary_interleaved=rotary_interleaved, alibi_slopes=None, num_splits=num_splits,
        return_softmax_lse=return_softmax_lse, page_table=page_table,
        attention_chunk=attention_chunk, q_descale=q_descale, k_descale=k_descale,
        v_descale=v_descale, pack_gqa=pack_gqa, sm_margin=sm_margin,
        scheduler_metadata=scheduler_metadata, cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new, max_seqlen_q=max_seqlen_q,
        rotary_seqlens=rotary_seqlens)


flash_attn_combine = _fa2.flash_attn_combine


def get_scheduler_metadata(*args, **kw):
    """У FA-3 это предпосчёт расписания под конкретную форму, который потом передают обратно
    как scheduler_metadata. У нас расписание считает лаунчер на каждом запуске (это единицы
    микросекунд), кэшировать нечего. Возвращаем None — FA-3 такой аргумент трактует как
    «считай сам», так что вызывающий работает без правок."""
    return None
