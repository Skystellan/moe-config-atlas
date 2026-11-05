# MoE Config Atlas

2025 年以来开源 MoE 大模型的稀疏度与形状统计。网站：<https://skystellan.github.io/moe-config-atlas/>

每个模型的专家数 E、top-k、共享专家数、隐藏维度、专家宽度、层数和注意力布局都直接从 Hugging Face 上的 `config.json` 解析，派生出路由稀疏度 E/k、专家激活比、参数激活比和专家粒度 G，并按厂家和大中小三档统计。

网站分四部分：

- **稀疏度**：E/k、专家激活比、参数激活比、专家粒度。
- **形状与注意力**：宽深比 d/L、注意力宽度比（query 头数 × 每头 QK 维度 / d_model）、Q/KV、RoPE 维度占比、全局注意力层占比、长上下文每 token 的 KV cache、最大上下文。
- **结构特征**：29 项，分五组。注意力（线性 / 滑窗 / 稀疏注意力、MLA、输出门、sink、QK-norm、K=V、跨层 KV、索引复用）、位置编码（部分 RoPE、NoPE）、残差与归一化（超连接 HC / mHC / iHC、AttnRes、残差门、后置 norm）、MoE 与 FFN（共享专家及其门、LatentMoE、零计算专家、路由打分函数、专家偏置、分组路由、哈希路由、FFN 激活、激活限幅）、输出与记忆（MTP、softcap、N-gram 记忆）。有三种看法：逐个模型的结构指纹矩阵（也可按厂家或档位汇总）、最多 4 个模型的并排对比（选择写进网址）、按季度的采用率。
- **前沿技术追踪**：`techniques.yaml` 里的 2024 年以来的结构和训练改动，每条有最早出处、独立实验测到的效果（主要来自 Pavlichenko 2026 的 1,129 次消融和 Marin 的训练记录）和收录模型里的采用情况。

## 怎么更新

- **数据刷新**：`pages.yml` 每周一和每次推送到 `main` 时重新拉取所有 `config.json`，有变化就提交 `site/data/` 并重新部署。
- **发现新模型**：`discover.yml` 每周一扫描 `discover.yaml` 里列出的厂家组织。MoE 形状和已收录模型相同的仓库会加进对应条目的 `variants`，新形状会作为新条目加进 `models.yaml`，然后开一个 PR。合并即收录，不想要的条目在 PR 里删掉即可（`discover_state.json` 会记住，下次不再提）。

## 手动收录一个模型

在 `models.yaml` 里加一条：

```yaml
- vendor: Qwen
  name: Qwen3-235B-A22B
  repo: Qwen/Qwen3-235B-A22B        # 解析这个仓库的 config.json
  variants: [Qwen/Qwen3-235B-A22B-Instruct-2507]   # 可选：MoE 配置相同的后续版本
  released: 2025-04                 # 首次上传的年月
  total: 235                        # 总参数（B），取自模型卡
  active: 22                        # 激活参数（B）
  override: {E: 128}                # 可选：config.json 读不到时手填
  notes: 可选的脚注
```

解析器在 `scripts/hfconfig.py`，遇到新的字段写法时在 `ALIASES` 里加别名即可。config 里不写、只在建模代码里体现的结构特征（比如 Qwen3 的 QK-norm、gpt-oss 的 attention sink）按 `model_type` 记在 `ARCH` 里；新架构上线时对照 transformers 或仓库里的 `modeling_*.py` 补一行。

仓库需要申请访问、Actions 读不到 `config.json` 时，可以把 config 存成 `configs/<org>__<repo>.json`（比如取自 transformers 里对应 `*Config` 的默认值），并在 `_source` 里写明出处；构建时优先读这里的文件，网页的详情里会注明来源。

## 追踪一项新技术

在 `techniques.yaml` 里加一条，字段说明在文件开头。能从 config 识别的写 `feature`（以及可选的 `match`），采用数会自动统计；config 看不出的（优化器、损失函数）在 `models` 里逐个列出采用的型号和出处链接。`build.py` 会检查分类、日期格式、出处和型号名。

## 本地运行

```bash
pip install -r requirements.txt
python scripts/build.py          # 生成 site/data/ 下的 models.json、models.csv 和 techniques.json
python -m http.server -d site    # 打开 http://localhost:8000
```

读需要申请访问的仓库时设置环境变量 `HF_TOKEN`（在仓库 Settings → Secrets 里配置同名 secret，Actions 也会用到）。
