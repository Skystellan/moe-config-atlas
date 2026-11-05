"""Normalize a Hugging Face config.json into a common set of MoE fields.

Vendors spell the same thing many ways (num_experts / n_routed_experts /
num_local_experts / moe_num_experts ...), nest the language model under
text_config / llm_config, and sometimes give per-layer lists. This module
hides all of that behind `parse(config) -> dict`.
"""
from collections import Counter
from fractions import Fraction
from statistics import mean

NESTED = ("text_config", "llm_config", "language_config", "thinker_config")

ALIASES = {
    "E": ["n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"],
    "k": ["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "moe_topk",
          "top_k_experts", "moe_k", "experts_per_token"],
    "s": ["n_shared_experts", "num_shared_experts", "moe_num_shared_experts",
          "num_shared_expert"],
    "d_model": ["hidden_size", "dim", "d_model"],
    "layers": ["num_hidden_layers", "num_layers", "n_layers"],
    "d_expert": ["expert_hidden_dim", "moe_intermediate_size", "expert_ffn_hidden_size",
                 "intermediate_size"],
    "d_shared": ["shared_expert_intermediate_size", "moe_shared_expert_intermediate_size",
                 "share_expert_dim", "shared_intermediate_size"],
    "latent": ["moe_latent_size", "routed_expert_hidden_size"],
    "zero_experts": ["zero_expert_num"],
    "vocab": ["vocab_size"],
}


def text_config(cfg):
    """Return the language-model part of a (possibly multimodal) config."""
    for key in NESTED:
        sub = cfg.get(key)
        if isinstance(sub, dict) and sub:
            return {**{k: v for k, v in cfg.items() if k not in NESTED}, **sub}
    if isinstance(cfg.get("moe"), dict):  # Mistral params.json
        return {**cfg, **cfg["moe"], "d_expert_moe": cfg["moe"].get("expert_hidden_dim")}
    return cfg


def _scalar(v):
    """Collapse per-layer lists to a representative value; flag variability."""
    if isinstance(v, list):
        nums = [x for x in v if isinstance(x, (int, float)) and x]
        if not nums:
            return None, False
        if len(set(nums)) == 1:
            return nums[0], False
        return Counter(nums).most_common(1)[0][0], True
    return v, False


def _get(c, field):
    for key in ALIASES[field]:
        if c.get(key) is not None:
            return _scalar(c[key])
    return None, False


# Token-mixer layer types, grouped by how their state grows with context.
GLOBAL = {"full", "MLA", "attention", "DSA", "compressed", "sparse"}  # cache grows with every token
LOCAL = {"sliding", "chunked"}  # bounded by the window or chunk
LINEAR = {"linear", "GDN", "KDA", "Lightning", "mamba", "conv"}  # constant-size recurrent state
LINEAR_NAMES = {"mamba": "Mamba-2", "conv": "门控卷积"}


def _summary(counts):
    counts = {k: v for k, v in counts.items() if v}
    if len(counts) == 1:
        return next(iter(counts))
    return " + ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


def _mixers(c, n_layers):
    """Count token-mixer layers by type, e.g. Counter(linear=36, full=12)."""
    counts = _mixer_counts(c, n_layers)
    # An indexer that picks which keys each query reads turns the global layers sparse:
    # Step-5 (`sparse_config`, block-compressed CSA) and Qwen3.8 (`indexer_*`, QSA).
    sc = c.get("sparse_config")
    indexed = isinstance(sc, dict) and sc.get("enabled") and "full_attention" in (sc.get("apply_to_layer_types") or ["full_attention"])
    if (indexed or c.get("indexer_budget")) and counts.get("full"):
        counts["sparse"] += counts.pop("full")
    return counts


def _mixer_counts(c, n_layers):
    full = "MLA" if c.get("kv_lora_rank") else "full"
    types = c.get("layer_types") or c.get("layers_block_type")
    pattern = c.get("hybrid_override_pattern")
    if pattern:
        names = {"M": "mamba", "*": "attention"}
        types = [names[ch] for ch in pattern if ch in names]
    if isinstance(types, list) and types:
        if c.get("layer_types") and n_layers:
            types = types[:n_layers]  # some configs append MTP layers
        types = [t for t in types if t not in ("moe", "mlp", "dense")]
        short = {"linear_attention": "linear", "full_attention": "full", "sliding_attention": "sliding",
                 "deepseek_sparse_attention": "DSA", "hybrid": "full", "hybrid_sliding": "sliding"}  # ZAYA1
        return Counter(short.get(t, t) for t in types)
    if isinstance(c.get("local_layer_ids"), list) and n_layers:  # Inkling: the other layers are global
        n_local = len(c["local_layer_ids"])
        return Counter(sliding=n_local, full=n_layers - n_local)
    if isinstance(c.get("gqa_layers"), list) and isinstance(c.get("linear_attn_config"), dict) and n_layers:
        n_full = len(c["gqa_layers"])  # Solar Open 2: KDA everywhere else
        return Counter(KDA=n_layers - n_full, full=n_full)
    if isinstance(c.get("no_rope_layers"), list) and c.get("attention_chunk_size") and n_layers:
        # Llama 4: the RoPE layers attend within fixed chunks, the NoPE layers see the whole context
        nr = c["no_rope_layers"][:n_layers]
        return Counter(chunked=sum(1 for x in nr if x), full=sum(1 for x in nr if not x))
    if isinstance(c.get("hybrid_layer_pattern"), list):  # MiMo: 1 = sliding window
        hp = c["hybrid_layer_pattern"]
        return Counter(sliding=sum(1 for x in hp if x), full=sum(1 for x in hp if not x))
    if isinstance(c.get("linear_attention_pattern"), list):  # Youtu: 1 = KDA
        lp = c["linear_attention_pattern"][:n_layers or None]
        return Counter(KDA=sum(1 for x in lp if x), **{full: sum(1 for x in lp if not x)})
    if isinstance(c.get("attn_type_list"), list):  # MiniMax: 0 = lightning (linear)
        at = c["attn_type_list"]
        return Counter(linear=sum(1 for x in at if not x), full=sum(1 for x in at if x))
    if c.get("full_attention_interval") and n_layers:  # Qwen3-Next
        n_full = n_layers // c["full_attention_interval"]
        return Counter(linear=n_layers - n_full, full=n_full)
    lac = c.get("linear_attn_config")
    if isinstance(lac, dict) and lac.get("kda_layers") is not None:  # Kimi Linear / K3
        return Counter(KDA=len(lac["kda_layers"]), **{full: len(lac.get("full_attn_layers") or [])})
    if c.get("layer_group_size") and any(k.startswith("kda") for k in c) and n_layers:  # Ling 3
        n_full = n_layers // c["layer_group_size"]
        return Counter(KDA=n_layers - n_full, **{full: n_full})
    kind = full
    if any(a.lower().startswith("longcat") for a in c.get("architectures") or []) and n_layers:
        return Counter({kind if not c.get("index_topk") else "DSA": 2 * n_layers})  # two attention sublayers per layer
    if isinstance(c.get("compress_ratios"), list) and n_layers:
        # DeepSeek V4: 0 = sliding window only, otherwise sliding window + compressed KV (CSA / HCA).
        ratios = c["compress_ratios"][:n_layers]  # trailing entries are MTP layers
        return Counter(sliding=sum(1 for r in ratios if not r), compressed=sum(1 for r in ratios if r))
    if isinstance(c.get("sparse_attention_config"), dict) and c["sparse_attention_config"].get("use_sparse_attention"):
        freq = c["sparse_attention_config"].get("sparse_attention_freq")  # MiniMax M3: 0 = dense layer
        if isinstance(freq, list) and n_layers:
            freq = freq[:n_layers]
            return Counter(sparse=sum(1 for x in freq if x), **{full: sum(1 for x in freq if not x)})
        kind = "sparse"
    elif c.get("index_topk") or c.get("index_n_heads"):
        kind = "DSA"
    return Counter({kind: n_layers or 0})


def _kv_bytes_per_global_layer(c):
    """bf16 KV-cache bytes one global-attention layer stores per token."""
    if c.get("kv_lora_rank"):  # MLA caches the compressed latent plus the RoPE key
        return (c["kv_lora_rank"] + (c.get("qk_rope_head_dim") or 64)) * 2
    n_heads = c.get("num_attention_heads") or c.get("n_heads")
    d = c.get("hidden_size") or c.get("dim")
    if c.get("global_head_dim"):  # Gemma 4 global layers
        kv = c.get("num_global_key_value_heads") or c.get("num_key_value_heads")
        k_dim = c["global_head_dim"]
        v_dim = 0 if c.get("attention_k_eq_v") else k_dim
    else:
        kv = (c.get("num_attention_groups_full_attention") or c.get("num_key_value_heads")
              or c.get("n_kv_heads") or c.get("num_attention_groups") or n_heads)
        k_dim = (c.get("head_dim_full_attention") or c.get("head_dim") or c.get("attention_head_dim")
                 or (d // n_heads if d and n_heads else None))
        v_dim = c.get("v_head_dim") or k_dim
    if not kv or not k_dim:
        return None
    return kv * (k_dim + v_dim) * 2


def _compressed_kv_bytes(c, n_layers):
    """DeepSeek V4: one shared K=V entry per `ratio` tokens in each compressed layer."""
    ratios = [r for r in c["compress_ratios"][:n_layers] if r]
    if not ratios or min(ratios) < 4 or not c.get("head_dim"):
        return None  # V4.1 switched to a different encoding; don't guess
    return sum(c["head_dim"] * 2 / r for r in ratios)


def _attn_shape(c):
    """Query heads and per-head query/key width of the global attention layers."""
    heads = c.get("num_attention_heads") or c.get("n_heads")
    d = c.get("hidden_size") or c.get("dim")
    if c.get("global_head_dim"):  # Gemma 4 global layers are wider than its sliding ones
        qk = c["global_head_dim"]
    elif c.get("kv_lora_rank") and c.get("qk_nope_head_dim") is not None:  # MLA: no-RoPE part + RoPE part
        # (Youtu's Megatron-style qk_head_dim leaves the RoPE part out, so don't read it)
        qk = c["qk_nope_head_dim"] + (c.get("qk_rope_head_dim") or 0)
    else:
        qk = (c.get("head_dim_full_attention") or c.get("head_dim") or c.get("attention_head_dim")
              or (d // heads if d and heads else None))
    return heads, qk or None


def _rope_fraction(c, layer_types=None):
    """Share of each global-attention head's query/key dims that get RoPE (0 = NoPE)."""
    if c.get("mla_use_nope") or c.get("position_embedding_type") == "nope" or c.get("use_rope") is False:
        return 0.0  # Granite 4.0-H, Solar Open 2
    nope_ids = c.get("no_rope_layer_indices")
    if isinstance(nope_ids, list) and isinstance(layer_types, list):  # flint
        full = {i for i, t in enumerate(layer_types) if t == "full_attention"}
        if full and full <= set(nope_ids):
            return 0.0
    if isinstance(c.get("no_rope_layers"), list) and c.get("attention_chunk_size") and 0 in c["no_rope_layers"]:
        return 0.0  # Llama 4: 0 marks the global layers, which skip RoPE
    if isinstance(c.get("partial_rotary_factors"), list) and isinstance(layer_types, list):  # Step: per layer
        vals = {f for f, t in zip(c["partial_rotary_factors"], layer_types) if t == "full_attention"}
        if len(vals) == 1:
            return float(vals.pop())
    rp = c.get("rope_parameters") if isinstance(c.get("rope_parameters"), dict) else {}
    if "full_attention" in rp and rp["full_attention"] is None:  # Beam: no RoPE on the full layers
        return 0.0
    if isinstance(rp.get("full_attention"), dict):  # Gemma 4: per layer type
        return float(rp["full_attention"].get("partial_rotary_factor") or 1.0)
    if c.get("kv_lora_rank") and c.get("qk_rope_head_dim") is not None:  # MLA's decoupled RoPE key
        return c["qk_rope_head_dim"] / ((c.get("qk_nope_head_dim") or 0) + c["qk_rope_head_dim"])
    if c.get("qk_rope_head_dim") is not None and c.get("head_dim"):  # DeepSeek V4: RoPE on 64 of 512 dims
        return c["qk_rope_head_dim"] / c["head_dim"]
    f = c.get("partial_rotary_factor") or rp.get("partial_rotary_factor") or c.get("rotary_percent")
    if f:
        return float(f)
    if c.get("rotary_dim") and c.get("head_dim"):
        return c["rotary_dim"] / c["head_dim"]
    return 1.0


def _frac(x):
    f = Fraction(x).limit_denominator(16)
    return f"{f.numerator}/{f.denominator}"


def _limit(xs):
    """SwiGLU clamp given per layer: '7' when every layer has it, '7（2 层）' when only some do."""
    on = [x for x in xs if x]
    if not on:
        return False
    vals = sorted({float(x) for x in on})
    label = "/".join(f"{v:g}" for v in vals)
    return label if len(on) == len(xs) else f"{label}（{len(on)} 层）"


def _moe_layers(c, n_layers):
    if not n_layers:
        return None
    for key in ("mlp_layer_types",):
        if isinstance(c.get(key), list):
            return sum(t == "sparse" for t in c[key])
    if isinstance(c.get("layers_block_type"), list):
        return sum(t == "moe" for t in c["layers_block_type"])
    if c.get("hybrid_override_pattern"):
        return c["hybrid_override_pattern"].count("E")
    if isinstance(c.get("block_configs"), list):
        return sum(b.get("block_type") == "moe" for b in c["block_configs"])
    if isinstance(c.get("moe_layer_freq"), list):
        return sum(1 for x in c["moe_layer_freq"] if x)
    if isinstance(c.get("moe_layers"), list):  # Llama 4
        return len(c["moe_layers"])
    if c.get("num_dense_layers") is not None:  # Trinity, LFM2: the first layers are dense
        return n_layers - c["num_dense_layers"]
    if c.get("dense_mlp_idx") is not None:  # Inkling
        return n_layers - c["dense_mlp_idx"]
    if isinstance(c.get("moe_layers_enum"), str):
        return len([x for x in c["moe_layers_enum"].split(",") if x.strip()])
    if isinstance(c.get("moe_layer_list"), list):
        return len(c["moe_layer_list"])
    if c.get("moe_layer_start_index") is not None:
        start, end = c["moe_layer_start_index"], c.get("moe_layer_end_index", n_layers - 1)
        if end in (None, -1):
            end = n_layers - 1
        return len(range(start, end + 1, c.get("moe_layer_interval", 1) or 1))
    if c.get("mlp_only_layers"):
        return n_layers - len(c["mlp_only_layers"])
    first = c.get("first_k_dense_replace") or 0
    return n_layers - first


# Architecture facts that live in modeling code rather than config.json, checked
# against transformers 5.21 and the vendors' remote modeling files. A key here
# overrides what the config flags say; None means "not public".
ARCH = {
    "qwen3_moe": dict(qk_norm=True, router="softmax", expert_bias=False),
    "qwen3_next": dict(qk_norm=True, attn_gate=True, router="softmax", expert_bias=False, shared_gate=True, linear="GDN"),
    "qwen3_5_moe_text": dict(qk_norm=True, attn_gate=True, router="softmax", expert_bias=False, shared_gate=True, linear="GDN"),
    "qwen4_exp_text": dict(qk_norm=True, attn_gate=True, router="softmax", expert_bias=False, shared_gate=True, linear="GDN"),
    "glm4_moe": dict(router="sigmoid", expert_bias=True),
    "glm4_moe_lite": dict(router="sigmoid", expert_bias=True),
    "glm_moe_dsa": dict(expert_bias=True),
    "glm5_next_text": dict(expert_bias=True, linear="KDA"),
    "gemma4_text": dict(qk_norm=True, router="softmax", expert_bias=False, post_norm="三明治"),
    "gpt_oss": dict(attn_sink=True, router="softmax", expert_bias=False, ffn_act="SwiGLU-OAI"),
    "deepseek_v4": dict(qk_norm=True, attn_sink=True, k_eq_v=True),
    "deepseek_v41_text": dict(qk_norm=True, attn_sink=True, k_eq_v=True),  # no public modeling code; attention assumed as in V4
    "kimi_linear": dict(expert_bias=True),
    "bailing_hybrid": dict(qk_norm=False),  # use_qk_norm is set but unused by the modeling code
    "longcat": dict(router="softmax", expert_bias=True),
    "minimax_text_01": dict(router="softmax", expert_bias=False, linear="Lightning"),
    "minimax_m3_vl": dict(qk_norm=True, expert_bias=True),
    "mistral4": dict(router="softmax", expert_bias=False),
    # Nemotron-H attention layers apply no rotary embedding; the Mamba-2 layers carry position.
    "nemotron_h": dict(router="sigmoid", expert_bias=True, rope=0.0),
    "nemotron_h_puzzle": dict(router="sigmoid", expert_bias=True, rope=0.0),
    "step3_text": dict(router="softmax", expert_bias=False),
    "step5": dict(qk_norm=None),  # no public modeling code and no qk-norm flag
    "ernie4_5_moe": dict(router="softmax", expert_bias=True),
    "hunyuan_v1_moe": dict(router="softmax", expert_bias=False),
    "hy_v4": dict(router="sigmoid", expert_bias=True),
    "mimo_v2_flash": dict(expert_bias=True),
    "mimo_v2": dict(expert_bias=True),
    "youtu_vl_moe": dict(qk_norm=True),
    # Muse Spark: scaleless QK-norm, elementwise output gate, NoPE on the full-attention layers,
    # sandwich norms with a learned residual gate, sigmoid router with a correction bias.
    "muse_spark_text": dict(qk_norm=True, attn_gate="逐元素", router="sigmoid", expert_bias=True,
                            post_norm="三明治", res_gate=True, rope=0.0),
    # Llama 4: sigmoid top-1 router next to an always-on shared expert.
    "llama4_text": dict(s=1, router="sigmoid", expert_bias=False),
    # Inkling: learned relative-position logits instead of RoPE; the shared experts are weighted by the router.
    "inkling_mm_model": dict(qk_norm=True, router="sigmoid", expert_bias=True, shared_gate="路由加权",
                             rope=0.0, nope="相对偏置"),
    "afmoe": dict(qk_norm=True, attn_gate="逐元素", expert_bias=True, post_norm="三明治"),
    "beam": dict(qk_norm=True, attn_gate="逐元素", router="sigmoid", expert_bias=True, post_norm="三明治", len_scale=True),
    "cohere2_moe": dict(expert_bias=False),
    "exaone_moe": dict(qk_norm=True, expert_bias=True),
    "axk1": dict(expert_bias=False, post_norm="仅 FFN"),  # an extra RMSNorm on the MoE block output only
    "axk2": dict(attn_gate="逐元素", expert_bias=True),
    "laguna": dict(qk_norm=True, router="sigmoid", expert_bias=True),  # softplus output gate
    "lfm2_moe": dict(qk_norm=True, router="sigmoid"),
    "granitemoehybrid": dict(router="softmax", expert_bias=False),
    "granitemoe_swa": dict(attn_sink=True, router="softmax", expert_bias=False),  # sink applied as LSE scaling
    # ZAYA1: L2 QK-norm, MLP router whose state carries across layers, an extra skip expert, learned
    # per-channel scales on the residual and on each sublayer output.
    "zaya": dict(qk_norm=True, router="softmax", expert_bias=True, zero_expert=1, res_gate="逐通道缩放"),
    "solar_open": dict(qk_norm=False, router="sigmoid", expert_bias=True),
    # Solar Open 2 ships no modeling code: the checkpoint has e_score_correction_bias as Solar Open does,
    # no q/k norm weights, and an elementwise g_proj on the GQA layers.
    "solar_open2": dict(qk_norm=False, attn_gate="逐元素", router="sigmoid", expert_bias=True),
    "mellum": dict(qk_norm=True, router="softmax", expert_bias=False),
    "kolibri1": dict(qk_norm=True, router="sigmoid", expert_bias=True, post_norm="三明治"),
    "flint": dict(qk_norm=True, expert_bias=False),
}

# These put RoPE only on the sliding-window layers whenever a model has both kinds (global NoPE).
NOPE_GLOBAL_IN_HYBRID = {"afmoe", "cohere2_moe", "exaone_moe", "kolibri1"}


FEATURES = ("linear_attn", "sparse_attn", "attn_gate", "attn_sink", "qk_norm", "k_eq_v", "kv_share", "index_share",
            "attn_conv", "partial_rope", "nope", "len_scale", "hyper_conn", "attn_res", "res_gate", "gated_norm",
            "post_norm", "parallel", "softcap", "ngram_mem", "router", "expert_bias", "group_routing", "hash_route",
            "zero_expert", "shared_gate", "ffn_act", "act_clamp")


def family(c):
    if c.get("model_type"):
        return c["model_type"]
    arch = ((c.get("architectures") or [""])[0]).lower()
    if arch.startswith("longcat"):
        return "longcat"
    if c.get("d_expert_moe"):  # Mistral params.json; Large 3 shares Small 4's MoE block
        return "mistral4"
    return None


def _flag(c, *keys):
    return any(c.get(k) not in (None, False, 0, "") for k in keys)


def features(c, mixers):
    """Structural features that don't reduce to a single number."""
    fam = family(c)
    arch = ARCH.get(fam, {})
    f = {}
    kinds = {k for k, v in mixers.items() if v}
    linear = kinds & LINEAR
    kind = ("mamba" if "mamba" in linear else sorted(linear)[0]) if linear else None
    f["linear_attn"] = LINEAR_NAMES.get(kind, kind) if kind else False
    sc = c.get("sparse_config") if isinstance(c.get("sparse_config"), dict) else {}
    if "DSA" in kinds:
        f["sparse_attn"] = "DSA"
    elif "compressed" in kinds:
        ratios = {r for r in c.get("compress_ratios") or [] if r}
        f["sparse_attn"] = "CSA+HCA" if ratios <= {4, 128} else "压缩"
    elif "sparse" in kinds:
        f["sparse_attn"] = ("QSA" if c.get("indexer_budget")
                            else "CSA" if str(sc.get("compression_method", "")).startswith("csa") else "块稀疏")
    else:
        f["sparse_attn"] = False
    gate = _flag(c, "attn_output_gate", "use_head_wise_attn_gate", "mla_use_output_gate", "gated_mla",
                 "attention_output_gate", "gated_attention_proj_granularity_type", "use_gqa_gate", "gating")
    granularity = (c.get("gating_type") or c.get("mla_output_gate_type")
                   or ("head_wise" if c.get("use_head_wise_attn_gate") else None))
    if _flag(c, "gating"):  # Laguna: True / "per-head" / "per-element" (a softplus gate)
        granularity = "elementwise" if c["gating"] == "per-element" else "head_wise"
    f["attn_gate"] = {"head_wise": "逐头", "elementwise": "逐元素"}.get(granularity, True) if gate else False
    f["attn_sink"] = (_flag(c, "learnable_sink", "add_swa_attention_sink_bias", "add_full_attention_sink_bias", "sink")
                      or c.get("softmax_type") == "learnable")
    f["qk_norm"] = _flag(c, "use_qk_norm", "qk_norm", "qk_layernorm", "qk_norm_type")
    if f["qk_norm"] and isinstance(c.get("no_rope_layers"), list):  # Llama 4 Scout: L2 norm on the RoPE layers only
        f["qk_norm"] = "仅局部层"
    f["k_eq_v"] = "全局层" if c.get("attention_k_eq_v") else False
    if c.get("kv_source_layer_ids"):  # DeepSeek V4.1: later layers read KV from the nearest source layer
        f["kv_share"] = f"{len(c['kv_source_layer_ids'])} 个源层"
    elif c.get("num_kv_shared_layers"):
        f["kv_share"] = f"后 {c['num_kv_shared_layers']} 层"
    elif c.get("use_cla"):
        f["kv_share"] = f"CLA×{c.get('cla_share_factor') or 2}"
    else:
        f["kv_share"] = False
    it = c.get("indexer_types")
    if isinstance(it, list) and "shared" in it:  # IndexShare: reuse an earlier layer's top-k picks
        f["index_share"] = f"{it.count('shared')}/{len(it)} 层"
    elif c.get("index_source_layer_ids"):
        f["index_share"] = f"{len(c['index_source_layer_ids'])} 个源层"
    elif (c.get("cli_factor") or 1) > 1:  # LongCat-2.0 cross-layer indexing
        f["index_share"] = f"每 {c['cli_factor']} 层"
    else:
        f["index_share"] = False
    if c.get("use_sconv"):  # Inkling: 4-tap causal convs on K and V and on both sublayer outputs
        f["attn_conv"] = "K、V、输出"
    elif c.get("cca_time0") or c.get("cca"):  # ZAYA1: Compressed Convolutional Attention mixes Q and K
        f["attn_conv"] = "Q、K（CCA）"
    else:
        f["attn_conv"] = False

    rope = arch.get("rope", _rope_fraction(c, c.get("layer_types")))
    if fam in NOPE_GLOBAL_IN_HYBRID and kinds & LOCAL and kinds & GLOBAL:
        rope = 0.0
    f["rope_frac"] = rope
    f["partial_rope"] = _frac(rope) if 0 < rope < 1 else False
    f["nope"] = rope == 0
    if fam == "cohere2_moe" and rope == 0 and c.get("prefix_dense_sliding_window_pattern") == 1:
        # the dense prefix layers keep RoPE even when they are global ones
        full = [i for i, t in enumerate(c.get("layer_types") or []) if t == "full_attention"]
        roped = [i for i in full if i < (c.get("first_k_dense_replace") or 0)]
        if roped:
            f["nope"] = f"{len(full) - len(roped)}/{len(full)} 层"
    # Query scaled up with log(position) on the global layers (Scalable-Softmax): Llama 4, Inkling
    f["len_scale"] = _flag(c, "attn_temperature_tuning", "log_scaling_n_floor")

    n_hc = c.get("hc_mult") or c.get("hc_count") or (4 if c.get("mhc") else None)
    hc = "mHC" if c.get("hc_sinkhorn_iters") else "iHC" if c.get("enable_ihc") else "HC"
    f["hyper_conn"] = f"{hc}×{n_hc}" if n_hc else False
    f["attn_res"] = c.get("attn_res_block_size") or False
    f["res_gate"] = _flag(c, "residual_gate_temperature")
    f["gated_norm"] = f"秩 {c['gated_norm_rank']}" if c.get("gated_norm") and c.get("gated_norm_rank") else bool(c.get("gated_norm"))
    f["post_norm"] = "Post-LN" if c.get("postnorm") else False
    f["parallel"] = _flag(c, "use_parallel_block", "parallel_attn", "use_parallel_residual")
    f["softcap"] = c.get("final_logit_softcapping") or c.get("attn_logit_softcapping") or False
    if c.get("engram_layer_ids"):
        f["ngram_mem"] = "Engram"
    elif c.get("ngram_size"):
        f["ngram_mem"] = "层内注入"  # hashed n-grams fed into the hyper-connection streams
    elif c.get("oe_vocab_size_ratio") or c.get("ngram_vocab_size_ratio"):
        f["ngram_mem"] = "嵌入层"
    else:
        f["ngram_mem"] = False
    router = (c.get("scoring_func") or c.get("moe_router_activation_func") or c.get("moe_router_activation")
              or c.get("score_func") or c.get("expert_selection_fn") or c.get("router_score_function")
              or c.get("gate_activation")
              or ("sigmoid" if c.get("moe_router_use_sigmoid") or c.get("use_moe_router_bias") else None))
    if fam == "bailing_moe":  # Ling 1.x routes with softmax, Ling 2.0 with sigmoid + expert bias
        router = "sigmoid" if c.get("moe_router_enable_expert_bias") else "softmax"
    f["router"] = router
    f["expert_bias"] = c.get("topk_method") == "noaux_tc" or _flag(
        c, "moe_router_enable_expert_bias", "use_routing_bias", "use_moe_router_bias",
        "moe_router_modality_aware_expert_bias", "use_expert_bias")
    if f["expert_bias"] and c.get("moe_router_expert_bias_update_method") == "quantile":
        f["expert_bias"] = "分位数"
    n_group = c.get("n_group") or c.get("num_expert_groups")
    f["group_routing"] = f"{n_group}/{c.get('topk_group') or c.get('num_expert_groups_per_tok')}" if (n_group or 0) > 1 else False
    f["hash_route"] = f"前 {c['num_hash_layers']} 层" if c.get("num_hash_layers") else False
    f["zero_expert"] = c.get("zero_expert_num") or False
    f["shared_gate"] = False  # only visible in modeling code, see ARCH

    act = (c.get("mlp_hidden_act") or c.get("hidden_act") or c.get("hidden_activation") or "silu").lower()
    f["ffn_act"] = ("SiTU" if act == "situ" else "ReLU²" if act == "relu2" else "GeGLU" if act.startswith("gelu")
                    else "SwiGLU-OAI" if act == "swigluoai" else "SwiGLU" if act == "silu" else act)
    if c.get("swiglu_limit"):
        f["act_clamp"] = f"{c['swiglu_limit']:g}"
    elif isinstance(c.get("swiglu_limits"), list):
        f["act_clamp"] = _limit(c["swiglu_limits"])
    elif isinstance(c.get("expert_swiglu_limit_list"), list):
        f["act_clamp"] = _limit(c["expert_swiglu_limit_list"])
    elif act == "situ" and c.get("activation_situ_beta"):  # beta·tanh(x/beta) bounds the gate softly
        f["act_clamp"] = f"软 β={c['activation_situ_beta']:g}"
    else:
        f["act_clamp"] = False
    for k, v in arch.items():
        if k not in ("linear", "rope"):
            f[k] = v
    return f


def parse(raw):
    c = text_config(raw)
    out, per_layer = {}, False
    for field in ALIASES:
        val, varies = _get(c, field)
        out[field] = val
        per_layer |= varies

    # Per-block configs (NVIDIA Puzzle): average over MoE blocks.
    blocks = [b for b in c.get("block_configs") or [] if b.get("block_type") == "moe"]
    if blocks:
        out["E"] = round(mean(b["n_routed_experts"] for b in blocks))
        out["k"] = Counter(b["num_experts_per_tok"] for b in blocks).most_common(1)[0][0]
        out["d_expert"] = Counter(b["moe_intermediate_size"] for b in blocks).most_common(1)[0][0]
        per_layer = per_layer or len({b["n_routed_experts"] for b in blocks}) > 1

    if c.get("d_expert_moe"):
        out["d_expert"] = c["d_expert_moe"]
    if family(c) == "muse_spark_text" and c.get("moe_hidden_size", 0) < (out["d_model"] or 0):
        out["latent"] = c["moe_hidden_size"]  # experts run at moe_hidden_size between two projections
    # Gemma 4 runs a dense MLP in parallel with the MoE block: count it as one shared expert.
    if c.get("enable_moe_block") and not out["s"]:
        out["s"] = 1
        out["d_shared"] = c.get("intermediate_size")
    # Shared expert given only as a width (Qwen3-Next style): one shared expert.
    if not out["s"] and out["d_shared"]:
        out["s"] = 1
    out["s"] = out["s"] or 0
    if out["E"] is not None and out["E"] <= 1:
        out["E"] = None  # dense model

    if not out["layers"] and isinstance(c.get("layers_block_type"), list):
        out["layers"] = len(c["layers_block_type"])
    out["moe_layers"] = _moe_layers(c, out["layers"])
    mixers = _mixers(c, out["layers"])
    label = ARCH.get(family(c), {}).get("linear")
    if label and "linear" in mixers:  # name the linear-attention kind, e.g. GDN or Lightning
        mixers[label] = mixers.pop("linear")
    out["attention"] = _summary(mixers)
    n_mix = sum(mixers.values())
    n_global = sum(v for k, v in mixers.items() if k in GLOBAL)
    out["mixer_layers"] = n_mix or None
    out["global_attn_frac"] = round(100 * n_global / n_mix, 1) if n_mix else None
    out["aspect"] = round(out["d_model"] / n_mix, 1) if n_mix and out["d_model"] else None
    if c.get("compress_ratios"):
        kv_bytes = _compressed_kv_bytes(c, out["layers"])
    else:
        per_layer_kv = _kv_bytes_per_global_layer(c)
        kv_bytes = n_global * per_layer_kv if per_layer_kv else None
    out["kv_kib_per_token"] = round(kv_bytes / 1024, 1) if kv_bytes else None
    local = sum(v for k, v in mixers.items() if k in LOCAL)
    has_window = local or c.get("compress_ratios")  # every DeepSeek V4 layer keeps a sliding window
    window = (c.get("attention_chunk_size") if mixers.get("chunked") else
              c.get("sliding_window") or c.get("sliding_window_size") or _scalar(c.get("sliding_windows"))[0])
    out["window"] = window if has_window else None
    out["mla"] = bool(c.get("kv_lora_rank"))
    out["head_dim"] = c.get("head_dim") or c.get("attention_head_dim")
    # Attention width of the global layers: query heads x per-head query/key dims, against d_model.
    heads, qk = _attn_shape(c)
    out["heads"], out["qk_dim"] = heads, qk
    out["attn_ratio"] = round(heads * qk / out["d_model"], 2) if heads and qk and out["d_model"] else None
    # KV heads of the global layers (Gemma 4 and Step use fewer there than in their local layers)
    kv = (c.get("num_global_key_value_heads") or c.get("num_attention_groups_full_attention")
          or c.get("num_key_value_heads") or c.get("n_kv_heads") or c.get("num_attention_groups"))
    out["kv_heads"] = None if out["mla"] else kv
    out["gqa"] = round(heads / kv, 1) if heads and kv and not out["mla"] else None
    out["context"] = c.get("max_position_embeddings") or c.get("max_position_embedding") or c.get("model_max_length")
    mtp_cfg = c["mtp_config"] if isinstance(c.get("mtp_config"), dict) else {}  # Inkling nests it
    out["mtp"] = next((src[k] for src in (c, mtp_cfg) for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers",
                                                                 "mtp_num_layers", "num_mtp_modules")
                       if isinstance(src.get(k), int)), 0)
    out["model_type"] = c.get("model_type") or raw.get("model_type")
    out.update(features(c, mixers))
    out["per_layer"] = per_layer
    return out


def is_moe(raw):
    try:
        p = parse(raw)
    except Exception:
        return False
    return bool(p["E"] and p["k"] and p["E"] > p["k"])
