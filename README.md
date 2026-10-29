# MoE Config Atlas

2025 年以来开源 MoE 大模型的稀疏度与形状统计。网站：<https://skystellan.github.io/moe-config-atlas/>

每个模型的专家数 E、top-k、共享专家数、隐藏维度、专家宽度、层数和注意力布局都直接从 Hugging Face 上的 `config.json` 解析，派生出路由稀疏度 E/k、专家激活比、参数激活比和专家粒度 G，并按厂家和大中小三档统计。

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

解析器在 `scripts/hfconfig.py`，遇到新的字段写法时在 `ALIASES` 里加别名即可。

## 本地运行

```bash
pip install -r requirements.txt
python scripts/build.py          # 生成 site/data/models.json 和 models.csv
python -m http.server -d site    # 打开 http://localhost:8000
```

读需要申请访问的仓库时设置环境变量 `HF_TOKEN`（在仓库 Settings → Secrets 里配置同名 secret，Actions 也会用到）。
