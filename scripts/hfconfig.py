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
    "E": ["n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts", "moe_num_primary_experts"],
    "k": ["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "moe_topk",
          "top_k_experts", "moe_k", "experts_per_token", "moe_num_active_primary_experts"],
    "s": ["n_shared_experts", "num_shared_experts", "moe_num_shared_experts",
          "num_shared_expert"],
    "d_model": ["hidden_size", "dim", "d_model"],
    "layers": ["num_hidden_layers", "num_layers", "n_layers"],
    "d_expert": ["expert_hidden_dim", "moe_intermediate_size", "expert_ffn_hidden_size", "moe_ffn_hidden_size",
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
            return _class_defaults({**{k: v for k, v in cfg.items() if k not in NESTED}, **sub})
    if isinstance(cfg.get("moe"), dict):  # Mistral params.json
        return {**cfg, **cfg["moe"], "d_expert_moe": cfg["moe"].get("expert_hidden_dim")}
    return _class_defaults(cfg)


def _class_defaults(c):
    """Fill in what a config.json leaves to its transformers config class, so that a checkpoint which writes
    the defaults out and one which doesn't parse the same. The Llama 4 base checkpoints give no_rope_layers
    as [] (which transformers reads as unset) and leave out moe_layers and attn_temperature_tuning."""
    n = c.get("num_hidden_layers")
    if c.get("model_type") != "llama4_text" or not isinstance(n, int):
        return c
    c = dict(c)
    if not c.get("no_rope_layers"):  # every 4th layer is a NoPE layer
        every = c.get("no_rope_layer_interval") or 4
        c["no_rope_layers"] = [int((i + 1) % every != 0) for i in range(n)]
    if c.get("moe_layers") is None:
        step = c.get("interleave_moe_layer_step") or 1
        c["moe_layers"] = list(range(step - 1, n, step))
    for key, default in (("attn_temperature_tuning", True), ("tie_word_embeddings", False)):
        if c.get(key) is None:
            c[key] = default
    return c


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
    return Counter(_mixer_seq(c, n_layers))


def _mixer_seq(c, n_layers):
    """Token-mixer type of each layer, in order, e.g. ['linear', 'linear', 'linear', 'full', ...]."""
    seq = _mixer_types(c, n_layers)
    # An indexer that picks which keys each query reads turns the global layers sparse:
    # Step-5 (`sparse_config`, block-compressed CSA) and Qwen3.8 (`indexer_*`, QSA).
    sc = c.get("sparse_config")
    indexed = isinstance(sc, dict) and sc.get("enabled") and "full_attention" in (sc.get("apply_to_layer_types") or ["full_attention"])
    if indexed or c.get("indexer_budget"):
        seq = ["sparse" if t == "full" else t for t in seq]
    return seq


def _mixer_types(c, n_layers):
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
                 "deepseek_sparse_attention": "DSA", "hybrid": "full", "hybrid_sliding": "sliding",  # ZAYA1
                 "indexed_attention": "DSA"}  # Mistral Large 4
        if c.get("index_topk") or c.get("index_n_heads"):  # dots3-note: the full layers carry a DSA indexer
            short["full_attention"] = "DSA"
        return [short.get(t, t) for t in types]
    if isinstance(c.get("full_attention_layers"), list) and c.get("linear_attention_type") and n_layers:
        full_ids = set(c["full_attention_layers"])  # GigaChat 3.5: 0-based; GatedDeltaNet everywhere else
        return [full if i in full_ids else "linear" for i in range(n_layers)]
    if isinstance(c.get("sliding_window_layout"), list):  # SmallThinker: 1 = sliding window
        return ["sliding" if x else full for x in c["sliding_window_layout"][:n_layers or None]]
    if isinstance(c.get("local_layer_ids"), list) and n_layers:  # Inkling: the other layers are global
        local = set(c["local_layer_ids"])
        return ["sliding" if i in local else "full" for i in range(n_layers)]
    if isinstance(c.get("gqa_layers"), list) and isinstance(c.get("linear_attn_config"), dict) and n_layers:
        gqa = set(c["gqa_layers"])  # Solar Open 2: KDA everywhere else
        return ["full" if i in gqa else "KDA" for i in range(n_layers)]
    if isinstance(c.get("no_rope_layers"), list) and c.get("attention_chunk_size") and n_layers:
        # Llama 4: the RoPE layers attend within fixed chunks, the NoPE layers see the whole context
        return ["chunked" if x else "full" for x in c["no_rope_layers"][:n_layers]]
    if isinstance(c.get("hybrid_layer_pattern"), list):  # MiMo: 1 = sliding window
        return ["sliding" if x else "full" for x in c["hybrid_layer_pattern"]]
    if isinstance(c.get("linear_attention_pattern"), list):  # Youtu: 1 = KDA
        return ["KDA" if x else full for x in c["linear_attention_pattern"][:n_layers or None]]
    if isinstance(c.get("attn_type_list"), list):  # MiniMax: 0 = lightning (linear)
        return ["full" if x else "linear" for x in c["attn_type_list"]]
    if c.get("full_attention_interval") and n_layers:  # Qwen3-Next: every k-th layer is full
        k = c["full_attention_interval"]
        return ["linear" if (i + 1) % k else "full" for i in range(n_layers)]
    lac = c.get("linear_attn_config")
    if isinstance(lac, dict) and lac.get("kda_layers") is not None:  # Kimi Linear / K3, 1-indexed
        kda, mla = set(lac["kda_layers"]), set(lac.get("full_attn_layers") or [])
        dsa = c["dsa_config"].get("dsa_layers") if isinstance(c.get("dsa_config"), dict) else None
        dsa = set(dsa) if isinstance(dsa, list) else set()  # K3.1 Pro: the MLA layers gain a top-k indexer
        return ["KDA" if i in kda else "DSA" if i in dsa else full for i in sorted(kda | mla)]
    if c.get("layer_group_size") and any(k.startswith("kda") for k in c) and n_layers:
        g = c["layer_group_size"]  # Ling 3: the last layer of each group, and any leftover layers, are MLA
        return [full if (i + 1) % g == 0 or i >= n_layers // g * g else "KDA" for i in range(n_layers)]
    if c.get("layer_group_size") and str(c.get("model_type") or "").startswith("bailing") and n_layers:
        # Ring linear and Ling 2.5 / 2.6: Lightning Attention-2 except in the last layer of each group and any
        # leftover layers, which are softmax attention (MLA in Ling 2.5 / 2.6)
        g = c["layer_group_size"]
        return [full if (i + 1) % g == 0 or i >= n_layers // g * g else "linear" for i in range(n_layers)]
    if isinstance(c.get("moba_layer_freq"), list) and n_layers:  # Ring-mini-sparse: 1 = MoBA block-sparse attention
        return ["sparse" if x else full for x in c["moba_layer_freq"][:n_layers]]
    kind = full
    if any(a.lower().startswith("longcat") for a in c.get("architectures") or []) and n_layers:
        kind = kind if not c.get("index_topk") else "DSA"
        ssa = c.get("streaming_sparse_attention") if isinstance(c.get("streaming_sparse_attention"), dict) else {}
        types = str(ssa.get("layer_type") or "")
        if len(types) == 2 * n_layers:  # ZigZag: in a "1" sublayer each query sees only the first and latest tokens
            return ["sliding" if t == "1" else kind for t in types]
        return [kind] * (2 * n_layers)  # two attention sublayers per layer
    if isinstance(c.get("compress_ratios"), list) and n_layers:
        # DeepSeek V4: 0 = sliding window only, otherwise sliding window + compressed KV (CSA / HCA).
        # ERNIE 5.5 Flash marks its MLA + DSA layers with a negative ratio.
        ratios = c["compress_ratios"][:n_layers]  # trailing entries are MTP layers
        return ["sliding" if not r else "compressed" if r > 0 else "DSA" for r in ratios]
    if isinstance(c.get("sparse_attention_config"), dict) and c["sparse_attention_config"].get("use_sparse_attention"):
        freq = c["sparse_attention_config"].get("sparse_attention_freq")  # MiniMax M3: 0 = dense layer
        if isinstance(freq, list) and n_layers:
            return ["sparse" if x else full for x in freq[:n_layers]]
        kind = "sparse"
    elif c.get("index_topk") or c.get("index_n_heads"):
        kind = "DSA"
    return [kind] * (n_layers or 0)


# One character per token-mixer layer for the site's layer strips.
LAYOUT_CODE = {"full": "A", "attention": "A", "MLA": "X", "DSA": "D", "sparse": "S", "sliding": "W", "chunked": "U",
               "GDN": "G", "KDA": "K", "Lightning": "N", "mamba": "B", "conv": "V", "linear": "L"}


def _layout(c, seq, linear_label):
    """The mixer sequence as a string of LAYOUT_CODE characters; DeepSeek V4's compressed layers are C (CSA,
    a few tokens per entry) or H (HCA, 128 tokens per entry), by their compress ratio."""
    ratios = c.get("compress_ratios") if isinstance(c.get("compress_ratios"), list) else []
    out = []
    for i, t in enumerate(seq):
        if t == "compressed":
            r = ratios[i] if i < len(ratios) else None
            out.append("C" if r and r <= 16 else "H")
        else:
            out.append(LAYOUT_CODE.get(linear_label if t == "linear" and linear_label else t, "?"))
    return "".join(out) or None


def _dense_layers(c, n_layers, moe_layers):
    """0-based indices of the layers whose FFN is dense rather than MoE, when the config says so per layer.
    None when it can't be told, or when the MoE blocks aren't paired with the mixer layers (Nemotron)."""
    if not n_layers or c.get("hybrid_override_pattern") or isinstance(c.get("block_configs"), list):
        return None
    if isinstance(c.get("mlp_layer_types"), list):
        dense = [i for i, t in enumerate(c["mlp_layer_types"][:n_layers]) if t != "sparse"]
    elif isinstance(c.get("layers_block_type"), list):
        return None
    elif isinstance(c.get("moe_layer_freq"), list):
        dense = [i for i, x in enumerate(c["moe_layer_freq"][:n_layers]) if not x]
    elif isinstance(c.get("moe_layers"), list):  # Llama 4
        moe = set(c["moe_layers"])
        dense = [i for i in range(n_layers) if i not in moe]
    elif c.get("num_dense_layers") is not None:
        dense = list(range(c["num_dense_layers"]))
    elif c.get("dense_mlp_idx") is not None:
        dense = list(range(c["dense_mlp_idx"]))
    elif isinstance(c.get("moe_layers_enum"), str):
        moe = {int(x) for x in c["moe_layers_enum"].split(",") if x.strip()}
        dense = [i for i in range(n_layers) if i not in moe]
    elif isinstance(c.get("moe_layer_list"), list):
        moe = set(c["moe_layer_list"])
        dense = [i for i in range(n_layers) if i not in moe]
    elif c.get("moe_layer_start_index") is not None:
        start, end = c["moe_layer_start_index"], c.get("moe_layer_end_index", n_layers - 1)
        moe = set(range(start, (n_layers - 1 if end in (None, -1) else end) + 1, c.get("moe_layer_interval", 1) or 1))
        dense = [i for i in range(n_layers) if i not in moe]
    elif c.get("mlp_only_layers"):
        dense = sorted(c["mlp_only_layers"])
    else:
        dense = list(range(c.get("first_k_dense_replace") or 0))
    return dense if n_layers - len(dense) == moe_layers else None


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
    ratios = [r for r in c["compress_ratios"][:n_layers] if r and r > 0]
    n_mla = sum(1 for r in c["compress_ratios"][:n_layers] if r and r < 0)  # ERNIE 5.5 Flash: MLA + DSA layers
    if not ratios or min(ratios) < 4 or not c.get("head_dim"):
        return None  # V4.1 switched to a different encoding; don't guess
    mla = 0
    if n_mla:
        if not c.get("hybrid_mla_kv_lora_rank"):
            return None
        mla = n_mla * (c["hybrid_mla_kv_lora_rank"] + (c.get("hybrid_mla_qk_rope_head_dim") or 64)) * 2
    return sum(c["head_dim"] * 2 / r for r in ratios) + mla


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
    rl, sw = c.get("rope_layout"), c.get("sliding_window_layout")
    if isinstance(rl, list) and isinstance(sw, list):  # SmallThinker: 0 = NoPE layer, per layer
        on_global = {r for r, s in zip(rl, sw) if not s}
        if on_global == {0}:
            return 0.0
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
    rot = c.get("rotary_dim") or c.get("rope_head_dim")  # K2-Horizon: rope_head_dim
    if rot and c.get("head_dim"):
        return rot / c["head_dim"]
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
    "qwen3_next": dict(qk_norm=True, attn_gate="逐元素", router="softmax", expert_bias=False, shared_gate=True, linear="GDN"),
    "qwen3_5_moe_text": dict(qk_norm=True, attn_gate="逐元素", router="softmax", expert_bias=False, shared_gate=True, linear="GDN"),
    "qwen4_exp_text": dict(qk_norm=True, attn_gate="逐元素", router="softmax", expert_bias=False, shared_gate=True, linear="GDN"),
    "glm4_moe": dict(router="sigmoid", expert_bias=True),
    "glm4_moe_lite": dict(router="sigmoid", expert_bias=True),
    "glm_moe_dsa": dict(expert_bias=True),
    "glm5_next_text": dict(expert_bias=True, linear="KDA"),
    "gemma4_text": dict(qk_norm=True, router="softmax", expert_bias=False, post_norm="三明治"),
    "gpt_oss": dict(attn_sink=True, router="softmax", expert_bias=False, ffn_act="SwiGLU-OAI"),
    "deepseek_v4": dict(qk_norm=True, attn_sink=True, k_eq_v=True),
    "deepseek_v41_text": dict(qk_norm=True, attn_sink=True, k_eq_v=True),  # no public modeling code; attention assumed as in V4
    "kimi_linear": dict(expert_bias=True),
    # Ling 3 (KDA + MLA) and Ling 2.5 / 2.6 (Lightning + MLA); qk_norm is set in features()
    "bailing_hybrid": dict(linear="Lightning"),
    # Ring linear: Lightning Attention-2 (a fixed decay per head) in all but the last layer of each group
    "bailing_moe_linear": dict(linear="Lightning"),
    # Ming flash omni (inclusionAI/Ming, modeling_bailing_moe_v2.py): q/k RMSNorm in every layer, whatever the
    # config says; sigmoid gates with an expert bias, one each for text, image and audio tokens (MultiRouter)
    "bailing_moe_v2": dict(qk_norm=True, router="sigmoid", expert_bias=True),
    # Ling 3.0 flash VL: use_qk_norm is set, but the checkpoint has no q/k norm weights, as in Ling 3.0
    "bailing_moe_v3_vl": dict(qk_norm=False),
    "longcat": dict(router="softmax", expert_bias=True),
    "minimax_text_01": dict(router="softmax", expert_bias=False, linear="Lightning"),
    "minimax_m1": dict(router="softmax", expert_bias=False, linear="Lightning"),  # MiniMax-Text-01's network
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
    "dots1": dict(qk_norm=True),
    # Mistral Large 4: softmax router without a bias; every layer scales queries by a learned per-head
    # a·log(pos+1)+b, and RoPE frequencies are learned (LeRoPE).
    "mistral_large4_text": dict(qk_norm=False, router="softmax", expert_bias=False, len_scale="逐头可学习"),
    # dots3-note: MLA with a per-head sigmoid gate; only the shared RoPE key is normalized, not each head's Q/K.
    "dots3_note": dict(qk_norm=False),
    "sarvam_mla": dict(qk_norm=False, router="sigmoid"),  # use_qk_norm is set but the MLA code never reads it
    # Instella-MoE: gated MLA (sigmoid of a full-width projection); qk_layernorm only names the latent norms.
    "instella_moe": dict(qk_norm=False, attn_gate="逐元素"),
    "mixtral": dict(router="softmax", expert_bias=False),
    "param1MoE": dict(router="softmax", expert_bias=False),  # Param-1: Mixtral's MoE block
    # GigaChat 3.5: GatedDeltaNet layers with MLA every 4th layer; a sigmoid gate over every MLA output channel;
    # norms before and after each sublayer (layernorm_type pre_post), each followed by a rank-16 2·sigmoid gate
    # (ZeroCenteredGatedNorm); sigmoid router with e_score_correction_bias.
    "gigachat3_5": dict(attn_gate="逐元素", router="sigmoid", expert_bias=True, post_norm="三明治",
                        gated_norm="秩 16", linear="GDN"),
    # K2-Horizon: the gate's bias only enters the top-k choice. MoVA-36B also routes the value projection
    # of its MoE layers over 64 value experts (top-4), noted in models.yaml.
    "k2_horizon": dict(expert_bias=True),
    # Klear: sigmoid router with a selection-only expert bias; the routed and shared outputs are mixed by a
    # softmax over two input-dependent coefficients.
    "Klear": dict(qk_norm=True, router="sigmoid", expert_bias=True, shared_gate="softmax 混合"),
    # SmallThinker: ReGLU experts; the router reads the layer input before the attention block, so experts can
    # be prefetched while attention runs.
    "smallthinker": dict(qk_norm=False, expert_bias=False, ffn_act="ReGLU"),
    # Vision-language releases whose language model is one of the families above
    "qwen3_vl_moe_text": dict(qk_norm=True, router="softmax", expert_bias=False),
    "glm4v_moe_text": dict(router="sigmoid", expert_bias=True),
    "longcat_next": dict(router="softmax", expert_bias=True),  # LongCat-Flash-Lite's backbone
    "minimax": dict(router="softmax", expert_bias=False, linear="Lightning"),  # MiniMax-Text-01 / M1 in HF format
    "step3p5": dict(qk_norm=True),  # Step 3.7 Flash drops use_qk_norm from its config; the q/k norm weights remain
}

# These put RoPE only on the sliding-window layers whenever a model has both kinds (global NoPE).
NOPE_GLOBAL_IN_HYBRID = {"afmoe", "cohere2_moe", "exaone_moe", "kolibri1"}


FEATURES = ("linear_attn", "sparse_attn", "attn_gate", "attn_sink", "qk_norm", "k_eq_v", "kv_share", "index_share",
            "attn_conv", "partial_rope", "nope", "len_scale", "hyper_conn", "attn_res", "res_gate", "gated_norm",
            "post_norm", "parallel", "softcap", "ngram_mem", "router", "expert_bias", "group_routing", "hash_route",
            "zero_expert", "shared_gate", "ffn_act", "act_clamp")


def family(c):
    arch = ((c.get("architectures") or [""])[0]).lower()
    if arch.startswith("instellamoe"):  # model_type says deepseek_v3, but the attention and residuals differ
        return "instella_moe"
    if c.get("model_type"):
        return c["model_type"]
    if arch.startswith("smallthinker"):  # the 21B config has no model_type
        return "smallthinker"
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
    comp = None
    if "compressed" in kinds:
        ratios = {r for r in c.get("compress_ratios") or [] if r and r > 0}
        comp = {frozenset({4, 128}): "CSA+HCA", frozenset({128}): "HCA", frozenset({4}): "CSA"}.get(frozenset(ratios), "压缩")
    if "DSA" in kinds:
        # LongCat Sparse Attention: DSA plus a fixed sink and local window in the top-k budget
        dsa = "LSA" if c.get("index_local_tokens") else "DSA"
        f["sparse_attn"] = f"{dsa}+{comp}" if comp else dsa  # ERNIE 5.5 Flash: DSA+HCA
    elif comp:
        f["sparse_attn"] = comp
    elif "sparse" in kinds:
        f["sparse_attn"] = ("QSA" if c.get("indexer_budget") else "MoBA" if c.get("moba_topk")
                            else "CSA" if str(sc.get("compression_method", "")).startswith("csa") else "块稀疏")
    else:
        f["sparse_attn"] = False
    gate = _flag(c, "attn_output_gate", "use_head_wise_attn_gate", "mla_use_output_gate", "gated_mla",
                 "attention_output_gate", "gated_attention_proj_granularity_type", "use_gqa_gate", "gating",
                 "gated_attention", "attention_gate_type", "use_gated_attn", "attention_gate_func")
    granularity = (c.get("gating_type") or c.get("mla_output_gate_type")
                   or {"headwise": "head_wise"}.get(c.get("attention_gate_type"), c.get("attention_gate_type"))
                   or {"element_wise": "elementwise"}.get(c.get("gated_attention_proj_granularity_type"),
                                                         c.get("gated_attention_proj_granularity_type"))
                   or ("head_wise" if c.get("use_head_wise_attn_gate") else None)
                   or ("elementwise" if c.get("mla_use_output_gate") else None)  # Kimi K3: heads x v_head_dim
                   or ("elementwise" if c.get("attention_gate_func") else None))  # K2-Horizon: heads x head_dim
    if _flag(c, "gating"):  # Laguna: True / "per-head" / "per-element" (a softplus gate)
        granularity = "elementwise" if c["gating"] == "per-element" else "head_wise"
    f["attn_gate"] = {"head_wise": "逐头", "elementwise": "逐元素"}.get(granularity, True) if gate else False
    f["attn_sink"] = (_flag(c, "learnable_sink", "add_swa_attention_sink_bias", "add_full_attention_sink_bias", "sink")
                      or c.get("softmax_type") == "learnable")
    f["qk_norm"] = _flag(c, "use_qk_norm", "qk_norm", "qk_layernorm", "qk_norm_type", "query_key_norm")
    if f["qk_norm"] and isinstance(c.get("no_rope_layers"), list) and c.get("attention_chunk_size"):
        # Llama 4 Scout: L2 norm on the RoPE layers only
        f["qk_norm"] = "仅局部层"
    f["k_eq_v"] = "全局层" if c.get("attention_k_eq_v") else False
    if c.get("kv_source_layer_ids"):  # DeepSeek V4.1: later layers read KV from the nearest source layer
        f["kv_share"] = f"{len(c['kv_source_layer_ids'])} 个源层"
    elif isinstance(c.get("kv_source_layers"), list):  # Mistral Large 4: some layers read an earlier layer's KV
        n = sum(1 for i, src in enumerate(c["kv_source_layers"]) if src != i)
        f["kv_share"] = f"{n} 层复用" if n else False
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
    nr = c.get("no_rope_layers")
    if isinstance(nr, list) and not c.get("attention_chunk_size") and 0 in nr and rope:
        # Mistral Large 4: every 4th global layer skips RoPE
        f["nope"] = f"{nr.count(0)}/{len(nr)} 层"
    # Query scaled up with log(position) on the global layers (Scalable-Softmax): Llama 4, Inkling, and
    # Mistral Large 3 / Small 4, which reuse Llama 4's scaling (llama_4_scaling_beta)
    rp = c.get("rope_parameters") if isinstance(c.get("rope_parameters"), dict) else {}
    f["len_scale"] = _flag(c, "attn_temperature_tuning", "log_scaling_n_floor", "llama_4_scaling") or _flag(
        rp, "llama_4_scaling_beta")

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
              or c.get("score_func") or c.get("score_function") or c.get("expert_selection_fn")
              or c.get("router_score_function") or c.get("router_score_func")
              or c.get("gate_activation")
              or ("sigmoid" if c.get("moe_router_use_sigmoid") or c.get("use_moe_router_bias") else None))
    if fam == "bailing_moe":  # Ling 1.x routes with softmax, Ling 2.0 with sigmoid + expert bias
        router = "sigmoid" if c.get("moe_router_enable_expert_bias") else "softmax"
    if fam == "bailing_moe_linear":  # Ring-lite-linear keeps Ling-lite's softmax router and says nothing
        router = router or "softmax"
    if fam == "bailing_hybrid":
        # Ling 3 sets use_qk_norm but its KDA and MLA layers have no q/k norm; Ling 2.5 / 2.6 normalize q and k
        # per head in the Lightning layers only, not in MLA
        f["qk_norm"] = "仅线性层" if f["qk_norm"] and "Lightning" in kinds else False
    if fam == "smallthinker":  # top-k on the logits, then softmax over the k, or sigmoid renormalized
        router = "softmax" if c.get("moe_primary_router_apply_softmax") else "sigmoid"
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
    # Usually only visible in modeling code, see ARCH. Command A+ averages the routed and shared outputs.
    f["shared_gate"] = ("取平均" if c.get("shared_expert_combination_strategy") == "average"
                        and c.get("num_shared_experts") else False)

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


# Smaller design choices compared side by side on the site: the exact form of each gate, how routing
# weights are normalized and scaled, norm and scaling conventions, RoPE base and extension, biases and the
# release format. None means not public or not determinable.
DETAILS = ("gate_act", "gate_src", "sink_layers", "lin_decay", "lin_out", "router_norm", "router_scale",
           "norm_form", "qk_norm_kind", "emb_proc", "attn_scale", "res_scale", "logit_scale", "out_softmax",
           "rope_theta", "rope_ext", "bias_terms", "tied_emb", "modality")

# Details that live in modeling code, checked against the same sources as ARCH.
DETAIL_ARCH = {
    # RMSNorm that scales by (1 + w) with w starting at 0
    "qwen3_next": dict(norm_form="RMSNorm(1+w)"),
    "qwen3_5_moe_text": dict(norm_form="RMSNorm(1+w)", router_norm=True),
    "qwen4_exp_text": dict(norm_form="RMSNorm(1+w)", router_norm=True),
    "minimax_m3_vl": dict(norm_form="RMSNorm(1+w)", router_norm=True),
    # Muse Spark: (1 + w) pre-norms, a weightless norm on the embeddings and on Q/K, router and output logits
    # times output_multiplier
    "muse_spark_text": dict(norm_form="RMSNorm(1+w)", emb_proc="RMSNorm", router_norm=True),
    "inkling_mm_model": dict(emb_proc="RMSNorm", router_norm=True, tied_emb=False),
    "minimax_m2": dict(qk_norm_kind="整层", router_norm=True),  # one RMSNorm over all heads, OLMo style
    # Llama 4's "L2Norm" is a weightless RMSNorm; sigmoid of the top-1 logit scales the expert input
    "llama4_text": dict(qk_norm_kind="无权重", router_norm=False),
    "zaya": dict(qk_norm_kind="无权重，k 乘每头温度", router_norm=False),  # q, k scaled to norm sqrt(d)
    "deepseek_v4": dict(qk_norm_kind="q 逐头无权重，KV 带权重"),
    "deepseek_v41_text": dict(qk_norm_kind="q 逐头无权重，KV 带权重"),
    "gpt_oss": dict(router_norm=True),  # softmax over the top-k logits
    "granitemoehybrid": dict(router_norm=True),
    "granitemoe_swa": dict(router_norm=True),
    # Gemma 4: router input is a weightless RMSNorm times a learned scale and d^-1/2; per-expert output scale
    "gemma4_text": dict(router_norm=True, router_scale="1，逐专家可学习", emb_proc="×√d"),
    "ernie4_5_moe": dict(router_norm=True),
    # ERNIE 5.5 model card: each routed expert's weight is multiplied by a learnable scalar of its own
    "ernie5_5_moe": dict(router_scale="3，逐专家可学习"),
    "minimax_text_01": dict(router_norm=True, tied_emb=False),  # MiniMax-VL-01 drops tie_word_embeddings
    "minimax_m1": dict(router_norm=True, tied_emb=False),
    # Ming flash omni: the config class defaults routed_scaling_factor to 2.5, and 2.0's config.json leaves it out
    "bailing_moe_v2": dict(router_scale=2.5),
    "minimax": dict(router_norm=True, tied_emb=False),
    "qwen3_vl_moe_text": dict(router_norm=True),  # the top-k weights are always renormalized
    "bailing_hybrid": dict(router_norm=True),  # Ling 2.5 / 2.6 renormalize whenever k > 1; Ling 3 says so in config
    "longcat_next": dict(router_norm=False, tied_emb=False),
    "step3p5": dict(tied_emb=False),  # lm_head ships as its own tensor
    "mistral4": dict(router_norm=True),  # Mistral Large 3's params.json has no flag; it shares Small 4's MoE block
    "mistral_large4_text": dict(router_norm=True),
    "laguna": dict(router_norm=True),
    "sarvam_mla": dict(router_norm=True),
    "longcat": dict(router_norm=False, tied_emb=False),  # softmax weights times 6-9, not renormalized
    "step3_text": dict(router_norm=False, tied_emb=False),
    "step5": dict(tied_emb=False),
    "lfm2_moe": dict(tied_emb=True),
    "cohere2_moe": dict(tied_emb=True),
    "beam": dict(res_scale="按深度保方差"),  # sublayer output times 1/sqrt(i), residual times sqrt((i-1)/i)
    # Solar Open 2 ships no modeling code: the weights show an elementwise g_proj, nothing about its activation
    "solar_open2": dict(gate_act=None, gate_src=None),
    "mixtral": dict(router_norm=True),  # softmax over all experts, then the top-k weights are renormalized
    "param1MoE": dict(router_norm=True),
    "Klear": dict(router_scale=1),  # config.json sets routed_scaling_factor, but the modeling code never reads it
}

_SCALE_KEYS = ("routed_scaling_factor", "moe_routed_scaling_factor", "router_scaling_factor",
               "moe_router_scaling_factor", "route_scale", "routed_scale")
_NORM_KEYS = ("norm_topk_prob", "moe_renormalize", "route_norm", "norm_expert_weight")


def _rope(c):
    """(theta, extension) of the global attention layers' RoPE as shipped in the config."""
    rp = c.get("rope_parameters") if isinstance(c.get("rope_parameters"), dict) else {}
    if isinstance(rp.get("full_attention"), dict):  # per-layer-type RoPE (Gemma 4, Laguna, Mellum)
        rp = rp["full_attention"]
    elif isinstance(rp.get("hybrid"), dict):  # ZAYA1
        rp = rp["hybrid"]
    rs = next((x for x in (c.get("rope_scaling"), c.get("yarn"), rp) if isinstance(x, dict) and x), {})
    theta = rp.get("rope_theta") or c.get("rope_theta")
    if isinstance(theta, list):  # Step: one base per layer, the large one on the full-attention layers
        theta = max(theta)
    kind = str(rs.get("rope_type") or rs.get("type") or ("yarn" if c.get("yarn") else ""))
    factor = rs.get("factor") or 1
    if "yarn" in kind and factor > 1:
        ext = f"YaRN ×{factor:g}"
    elif kind == "llama3" and factor > 1:
        ext = f"Llama 3 式 ×{factor:g}"
    elif kind == "dynamic" and rs.get("alpha"):
        ext = f"NTK α={rs['alpha']:g}"  # Hunyuan: dynamic NTK with a fixed alpha
    else:
        ext = False
    return theta, ext


def details(c, raw, fam, f):
    arch = DETAIL_ARCH.get(fam, {})
    d = {k: None for k in DETAILS}
    pick = lambda key, default: arch[key] if key in arch else default

    # Attention output gate: activation and what it reads (the layer input x, or MLA's query latent)
    if f.get("attn_gate"):
        d["gate_act"] = pick("gate_act", c.get("gated_attn_act") or c.get("attention_gate_func")
                             or ("softplus" if _flag(c, "gating") else "sigmoid"))
        d["gate_src"] = pick("gate_src", "q 潜向量" if _flag(c, "gated_attn_use_q_lora", "attn_gate_fused") else "x")
    # Learnable sink: one extra logit per head in the softmax, which scales the output by sigmoid(lse - s)
    if f.get("attn_sink"):
        swa, full = c.get("add_swa_attention_sink_bias"), c.get("add_full_attention_sink_bias")
        d["sink_layers"] = ("仅滑窗层" if swa and full is False else
                            "全局层" if full and not swa else "所有注意力层")  # ERNIE 5.5 only flags its MLA layers

    # Linear-attention gates: the decay (forget) gate and the output gate
    lac = c.get("linear_attn_config") if isinstance(c.get("linear_attn_config"), dict) else {}
    kind = f.get("linear_attn")
    if kind == "GDN" or kind == "Mamba-2":
        d["lin_decay"] = "逐头标量"
        sig = c.get("output_gate_type") == "sigmoid" or "sigmoid" in str(c.get("linear_gating_type") or "")
        d["lin_out"] = ("SiLU，先门后 norm" if kind == "Mamba-2" else "sigmoid，先 norm" if sig else "SiLU，先 norm")
    elif kind == "KDA" and fam == "solar_open2":
        d["lin_decay"] = "逐通道（细节未公开）"
    elif kind == "KDA":
        bound = next((x for x in (lac.get("gate_lower_bound"), c.get("kda_lower_bound"),
                                  c.get("kda_gate_lower_bound")) if x is not None), None)
        full_decay = bool(c.get("no_kda_lora"))
        full_out = full_decay or _flag(lac, "use_full_rank_gate") or _flag(c, "use_full_rank_gate")
        d["lin_decay"] = "逐通道，" + ("满秩" if full_decay else "低秩") + (f"，下界 {bound:g}".replace("-", "−") if bound else "")
        d["lin_out"] = "sigmoid，先 norm" if full_out else "sigmoid 低秩，先 norm"
    elif kind == "Lightning":
        d["lin_decay"], d["lin_out"] = "固定（每头一个常数）", "sigmoid，先 norm"
    elif kind == "门控卷积":
        d["lin_decay"], d["lin_out"] = "无", "线性门 B、C"

    # Routing: are the top-k weights renormalized, and by what are they scaled
    if f.get("router"):
        norm = next((c[k] for k in _NORM_KEYS if c.get(k) is not None), arch.get("router_norm"))
        d["router_norm"] = bool(norm) if norm is not None else None
        scale = arch.get("router_scale") or next((c[k] for k in _SCALE_KEYS if c.get(k) is not None), 1)
        if _flag(c, "use_global_scale"):  # Inkling: route_scale times one learned scalar
            scale = f"{scale:g}，可学习"
        d["router_scale"] = scale

    # Norms
    ln = c.get("layer_norm_eps") and not c.get("rms_norm_eps") and fam == "cohere2_moe"
    d["norm_form"] = "LayerNorm" if ln else arch.get("norm_form", "RMSNorm")
    if str(c.get("norm_type") or "").startswith("ZeroCentered"):  # GigaChat 3.5
        d["norm_form"] = "RMSNorm(1+w)"
    if (c.get("layernorm_num_groups") or 1) > 1:  # K2-Horizon MoVA: each half of d_model normalized on its own
        d["norm_form"] = f"分组 RMSNorm（{c['layernorm_num_groups']} 组）"
    if f.get("qk_norm"):
        d["qk_norm_kind"] = arch.get("qk_norm_kind", "逐头")
        if c.get("qk_scale_factor"):  # Muse Spark: weightless QK-norm, then the query times a constant
            d["qk_norm_kind"] = f"无权重，q 乘 {c['qk_scale_factor']:.3g}"
    if c.get("embedding_multiplier"):
        d["emb_proc"] = f"×{c['embedding_multiplier']:g}（μP）"  # Granite
    else:
        d["emb_proc"] = arch.get("emb_proc", False)

    # Scales on the attention logits and values besides 1/sqrt(d); SSMax-style length scaling is a feature
    scales = []
    if c.get("attention_multiplier"):
        scales.append(f"μP 1/{round(1 / c['attention_multiplier'])}")  # Granite: 1/d_head instead of 1/sqrt(d_head)
    if c.get("qk_scale_factor"):
        scales.append(f"q×{c['qk_scale_factor']:.3g}")
    if c.get("attention_value_scale"):
        scales.append(f"V×{c['attention_value_scale']:g}")  # MiMo
    if _flag(c, "mla_scale_q_lora", "apply_mla_qkv_lora_rescale", "use_mla_scaling_factor"):
        scales.append("MLA ×√(d/r)")  # LongCat, dots3-note, GigaChat 3.5: q and k times sqrt(d_model / lora rank)
    d["attn_scale"] = scales

    # Residual adds that are not h + f(h) (hyper-connections and residual gates are features)
    alpha = c.get("layernorm_mlp_alpha") or c.get("mlp_alpha_factor") or 1  # the second is the HF-format name
    if alpha != 1:
        d["res_scale"] = f"DeepNorm α={alpha:.3g}"  # MiniMax-Text-01: Post-LN with the residual times (2N)^(1/4)
    elif c.get("residual_multiplier"):
        d["res_scale"] = f"×{c['residual_multiplier']:g}（μP）"  # Granite: h + m * f(h)
    else:
        d["res_scale"] = arch.get("res_scale", False)

    # Output logits: a fixed divisor or multiplier, and a softmax replacement (softcap is a feature)
    if c.get("logits_scaling"):
        d["logit_scale"] = f"÷{c['logits_scaling']:g}（μP）"  # Granite
    elif c.get("logits_mup_width_multiplier"):
        d["logit_scale"] = f"÷{c['logits_mup_width_multiplier']:g}（μP）"  # Inkling
    elif c.get("output_multiplier"):
        d["logit_scale"] = f"×{c['output_multiplier']:.3g}"  # Muse Spark, before the softcap
    else:
        d["logit_scale"] = False
    d["out_softmax"] = "MultiMax" if c.get("multimax") == "lm_head" else False  # ERNIE 5.5

    d["rope_theta"], d["rope_ext"] = _rope(c)
    if f.get("rope_frac") == 0:  # the global layers carry no RoPE, so their base and extension don't apply
        d["rope_theta"], d["rope_ext"] = None, None

    if c.get("attention_bias"):
        d["bias_terms"] = "全部线性层" if fam == "gpt_oss" else "QKV"
    else:
        d["bias_terms"] = False
    tie = next((x for x in (c.get("tie_word_embeddings"), raw.get("tie_word_embeddings"),
                            raw.get("tied_embeddings")) if x is not None), None)  # Mistral params.json
    d["tied_emb"] = arch.get("tied_emb") if tie is None else bool(tie)
    has = lambda *keys: any(isinstance(src.get(k), dict) for src in (raw, c) for k in keys)
    if raw.get("language_model_only") or c.get("language_model_only"):
        d["modality"] = "文本"  # Qwen-AgentWorld keeps Qwen3.6's vision_config but ships no vision tower
    elif has("audio_config", "sound_config", "speech_config", "whisper_config"):  # Nemotron Omni: sound_config
        d["modality"] = "图文音"
    elif has("vision_config", "visual_config", "vision_encoder") or isinstance(c.get("vision_n_layers"), int):
        d["modality"] = "图文"  # DeepSeek-V4-Flash-Vision-Exp gives its encoder as flat vision_* keys
    else:
        d["modality"] = "文本"
    return d


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
    seq = _mixer_seq(c, out["layers"])
    mixers = Counter(seq)
    label = ARCH.get(family(c), {}).get("linear")
    if label and "linear" in mixers:  # name the linear-attention kind, e.g. GDN or Lightning
        mixers[label] = mixers.pop("linear")
    out["layout"] = _layout(c, seq, label)
    # dense-FFN layers, drawn under the layer strip when each mixer layer has its own FFN
    out["dense_layers"] = _dense_layers(c, out["layers"], out["moe_layers"]) if len(seq) == out["layers"] else None
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
    ssa = c.get("streaming_sparse_attention") if isinstance(c.get("streaming_sparse_attention"), dict) else {}
    window = (c.get("attention_chunk_size") if mixers.get("chunked") else
              c.get("sliding_window") or c.get("sliding_window_size") or _scalar(c.get("sliding_windows"))[0]
              or ssa.get("recent_size"))  # LongCat ZigZag: the latest 896 tokens, besides the first 128
    out["window"] = window if has_window else None
    out["window_sink"] = ssa.get("sink_size") if has_window and ssa else None  # ZigZag also keeps the first 128 tokens
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
    out.update(details(c, raw, family(c), out))
    out["per_layer"] = per_layer
    return out


# Fields of parse() that don't make a different network: context length, RoPE base and extension, the class name,
# whether some value varies per layer, MTP / draft layers, which a release may ship or leave out, and head_dim as
# written in the config: qk_dim holds the width the heads really use, while head_dim may be nominal (GLM-5's MLA)
# or missing where it equals d_model / heads (Ling-lite before 1.5-2506).
NOT_ARCH = ("context", "rope_theta", "rope_ext", "model_type", "per_layer", "mtp", "head_dim")


def arch_signature(parsed):
    """What two checkpoints must share to count as one architecture: everything parse() reads except NOT_ARCH.
    The versions in one models.yaml row must all have the same signature."""
    return {k: v for k, v in parsed.items() if k not in NOT_ARCH}


def is_moe(raw):
    try:
        p = parse(raw)
    except Exception:
        return False
    return bool(p["E"] and p["k"] and p["E"] > p["k"])
