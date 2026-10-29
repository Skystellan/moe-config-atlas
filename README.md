# MoE Config Atlas

2025 年以来开源 MoE 大模型的稀疏度与形状统计。网站：<https://skystellan.github.io/moe-config-atlas/>

每个模型的专家数 E、top-k、共享专家数、隐藏维度、专家宽度、层数和注意力布局都直接从 Hugging Face 上的 `config.json` 解析，派生出路由稀疏度 E/k、专家激活比、参数激活比和专家粒度 G，并按厂家和大中小三档统计。

网站分三部分：

- **稀疏度**：E/k、专家激活比、参数激活比、专家粒度。
- **形状与注意力**：宽深比 d/L、全局注意力层占比、长上下文每 token 的 KV cache、最大上下文。
- **结构特征**：线性 / 滑窗 / 稀疏注意力、MLA、注意力输出门、attention sink、QK-norm、超连接（HC / mHC）、AttnRes、N-gram 记忆（Engram 等）、MTP、路由打分函数、专家偏置、分组路由、共享专家、LatentMoE，按季度看采用率，按厂家或档位看谁在用。

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

## 本地运行

```bash
pip install -r requirements.txt
python scripts/build.py          # 生成 site/data/models.json 和 models.csv
python -m http.server -d site    # 打开 http://localhost:8000
```

读需要申请访问的仓库时设置环境变量 `HF_TOKEN`（在仓库 Settings → Secrets 里配置同名 secret，Actions 也会用到）。
