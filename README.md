# CleanEM

CleanEM 是一个面向表格数据的 **Agentic 错误检测、自适应机制发现与可解释证据融合框架**。项目通过结合表格画像分析、LLM 生成的语义/格式/跨列约束规则、计划驱动的规范化候选、自适应错误机制发现以及基于主动学习（D-optimal）的证据权重校准，输出每个单元格的错误后验概率、统一可解释证据追踪（Explanation Traces）以及候选修复建议。

> 说明：本目录为独立版 CleanEM，移除了旧版硬编码的 `legacy_specialized` 检测器；关于不同机制模式与消融实验的详细说明，请参阅 [MECHANISM_MODES.md](MECHANISM_MODES.md)。

---

## 目录结构

```text
cleanem/
├── README.md                                  # 项目使用说明（本文件）
├── MECHANISM_MODES.md                         # 机制模式与消融实验参数说明
├── environment.yml                            # Conda 环境配置（默认环境名: cleaningllm）
├── pytest.ini                                 # Pytest 配置（默认测试目录: test/）
├── main.py                                    # 主命令行入口（支持 clean_em 与 dual 模式）
├── profiler.py                                # 基于 Pandas 的表/列画像与表面契约分析
├── agent.py                                   # LLM Agent（列语义角色、规则生成、跨列关系、规范化规划、机制命名）
├── cleanem_models.py                          # 核心数据结构（ValueRecord / CellRecord / EvidenceObservation / RepairCandidate）
├── cleanem_inference.py                       # 证据矩阵构建、D-optimal 主动查询选择与校准后验推断
├── adaptive_mechanism_discovery.py            # 无监督候选机制聚类、查询配额分配与主动标签验证
├── canonicalization.py                        # 计划驱动的 JSON-DSL 规范化执行与支持度验证
├── judge.py                                   # 规则评估（Violation Rate / Dual）与 Ground-Truth 评测
├── dual_types.py                              # Dual 模式（P_clean / P_dirty）数据结构
├── conflict_resolver.py                       # Dual 模式规则冲突消解
├── gap_resolver.py                            # Dual 模式灰区（Gap Zone）扩展与消解
├── modification_memory.py                     # Dual 模式规则迭代记忆与日志
├── validator.py                               # Dual 模式规则互斥性验证
├── core/                                      # 基础工具子模块
│   ├── deducer.py                             # 规则冲突拓扑分析
│   ├── llm_usage.py                           # OpenAI 兼容接口的 Token 消耗记录器
│   ├── pattern_explorer.py                    # 多轮 LLM 列模式探索器
│   ├── pattern_types.py                       # PatternSpec / PatternEntry 定义
│   └── utils.py                               # 安全求值字典与通用辅助函数
├── scripts/                                   # 运行脚本、数据对齐与离线评测脚本
│   ├── run_test.sh                            # 示例批量运行脚本（hospital / flights / beers）
│   ├── prepare_tableeg_experiment.py          # 将原始 TableEG 数据对齐为 CleanEM 输入格式
│   ├── evaluate_explanation_faithfulness.py   # 离线评测解释追踪的反事实忠实度（Faithfulness）
│   └── evaluate_tableeg_semantic_correctness.py # 评测检测、解释语义、修复与约束对齐指标
├── test/                                      # 单元测试目录（pytest / unittest）
│   ├── conftest.py
│   ├── test_adaptive_mechanism_discovery.py
│   ├── test_archetype_feature_namespaces.py
│   ├── test_canonicalization_alternatives.py
│   ├── test_corrected_value_standardization.py
│   ├── test_ground_truth_values_equal.py
│   ├── test_low_budget_precision_and_enum.py
│   ├── test_nested_budget_trajectory.py
│   └── test_relationship_explanations.py
└── data/                                      # 内置基准数据集与 TableEG 对齐数据
    ├── beers_* / beers_config.json            # Beers 数据集及按错误类型拆分的子集
    ├── billionaire_*                          # Billionaire 数据集
    ├── flights_* / flights_config.json        # Flights 数据集
    ├── hospital_* / hospital_config.json      # Hospital 数据集
    ├── movies_*                               # Movies 数据集
    ├── rayyan_*                               # Rayyan 数据集
    ├── tax* (50k / 100k / 150k / 200k / full) # Tax 大规模数据集
    └── tableeg_experiment_b/                  # 对齐后的 TableEG 基准（beers / flight / hospital / restaurant）
```

---

## 环境准备

### 1. 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate cleaningllm
```

### 2. 配置 LLM 服务

CleanEM 通过 OpenAI 兼容 API 调用大语言模型（默认面向本地部署的 vLLM 服务 `http://127.0.0.1:8000/v1`）：

- 若使用本地 vLLM 服务，先在对应端口启动 OpenAI 兼容服务；若未指定 `--model`，程序会自动通过 `/v1/models` 获取可用模型。
- 若使用远程 OpenAI 兼容服务，可通过 `--base_url`、`--model` 和环境变量 `OPENAI_API_KEY` 指定。

常用环境变量：

| 环境变量 | 说明 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI 兼容接口密钥 | `EMPTY` |
| `CLEANEM_LLM_SEED` | LLM 采样与部分随机过程的固定随机种子 | 未设置 |
| `CLEANEM_LLM_MAX_ATTEMPTS` | 单次 LLM 请求最大重试次数 | `4` |
| `CLEANEM_LLM_FAIL_FAST` | 设为 `1`/`true` 时遇到不可恢复的 LLM 错误立即中止 | 关闭 |
| `CLEANEM_USAGE_LOG` | LLM Token 消耗记录 JSONL 路径（优先级低于 `--llm_usage_log`） | `<output_dir>/llm_usage.jsonl` |

---

## 快速开始

### 1. 运行单个数据集（CleanEM 主模式）

```bash
python main.py \
  --mode clean_em \
  --dirty_csv data/beers_error-01.csv \
  --clean_csv data/beers_clean.csv \
  --output_dir results/clean_em_beers \
  --base_url http://127.0.0.1:8000/v1 \
  --active_label_budget 10
```

> 说明：
> - `--clean_csv` 用于在离线基准测试中提供主动查询的 Oracle 标签与最终 Precision / Recall / F1 评估；若仅做无标签运行，可设置 `--active_label_budget 0` 并置空 `--clean_csv ""`。
> - 也可直接运行内置批量示例脚本：
>   ```bash
>   bash scripts/run_test.sh
>   ```

### 2. 使用缓存加速重复实验与消融对比

在对比不同主动学习预算或机制消融时，建议复用 LLM 产出的规则/规范化/关系缓存以及嵌套查询轨迹缓存：

```bash
python main.py \
  --mode clean_em \
  --dirty_csv data/hospital_error-01.csv \
  --clean_csv data/hospital_clean.csv \
  --output_dir results/clean_em_hospital_b20 \
  --active_label_budget 20 \
  --rule_pool_cache results/cache/hospital_rules.json \
  --canonicalization_plan_cache results/cache/hospital_canon.json \
  --relationship_rule_cache results/cache/hospital_rel.json \
  --active_query_trajectory_cache results/cache/hospital_traj.json \
  --active_query_checkpoints 0,4,8,10,20,40
```

---

## 常用命令行参数

可通过 `python main.py --help` 查看全部参数。以下列出核心参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--mode` | `clean_em` | 流水线模式：`clean_em`（主框架）或 `dual`（P_clean / P_dirty 双规则精炼） |
| `--dirty_csv` | `data/beers_error-01.csv` | 待检测的脏表 CSV 路径 |
| `--clean_csv` | `data/beers_clean.csv` | 可选的干净表 CSV，用于 Oracle 标签模拟与评测 |
| `--output_dir` | `results/agentic_error_detector` | 输出目录 |
| `--base_url` | `http://127.0.0.1:8000/v1` | OpenAI 兼容 LLM 服务地址 |
| `--model` | `None` | 可选模型名覆盖（默认自动从服务获取） |
| `--max_workers` | `8` | 按列并行调用 LLM 的最大线程数 |
| `--score_threshold` | `0.5` | 单元格后验概率判定为错误的默认阈值 |
| `--active_label_budget` | `10` | 用于证据校准与机制验证的主动标注预算（单元格数） |
| `--active_query_strategy` | `d_optimal` | 主动查询策略：`d_optimal` 或 `random` |
| `--adaptive_mechanism_query_fraction` | `0.40` | 固定主动标注预算中分配给未验证候选机制的比例 |
| `--active_query_clean_proxy_fraction` | `0.30` | 固定主动标注预算中保留给疑似干净样本（clean proxy）的比例 |
| `--mechanism_mode` | `adaptive_agent` | 机制发现模式：`adaptive_agent`（完整自适应）或 `agent_rules_only`（关闭自适应发现） |
| `--enable_corrected_value_standardization` | `False` | 利用主动查询返回的纠正值学习格式契约并参与第二轮检测 |
| `--rule_pool_cache` | `None` | 列语义与生成规则池的 JSON 缓存路径 |
| `--canonicalization_plan_cache` | `None` | LLM 规范化计划 JSON 缓存路径 |
| `--relationship_rule_cache` | `None` | LLM 跨列关系规则 JSON 缓存路径 |
| `--active_query_cache` | `None` | 主动查询单元格集合 JSON 缓存路径 |
| `--active_query_trajectory_cache` | `None` | 跨预算复用的嵌套主动查询轨迹缓存路径 |
| `--llm_usage_log` | `<output_dir>/llm_usage.jsonl` | 单次 LLM 请求的 Token 用量日志路径 |

### 常见消融开关

更多细节见 [MECHANISM_MODES.md](MECHANISM_MODES.md)：

- **关闭自适应机制发现**：`--mechanism_mode agent_rules_only` 或 `--disable_adaptive_mechanism_discovery`
- **关闭 Agent 生成规则**：`--disable_agent_rules`
- **关闭表级 LLM 函数依赖/关系规则**：`--disable_llm_relationship_rules`
- **关闭规范化证据进入推断**：`--disable_canonicalization_evidence`
- **关闭证据权重学习（等权基线）**：`--disable_evidence_weight_learning`
- **关闭基于列语义角色（Archetype）的证据门控**：`--disable_evidence_gating`
- **随机主动查询基线**：`--active_query_strategy random --disable_adaptive_mechanism_query_replacement`

---

## 输出文件说明

运行 `clean_em` 模式后，`--output_dir` 下会生成以下主要文件：

| 文件名模式 | 说明 |
| --- | --- |
| `<timestamp>_<dataset>_clean_em_scores.csv` | 每个单元格的先验、后验错误概率、硬标签与最终预测结果 |
| `<timestamp>_<dataset>_explanation_traces.jsonl` | 所有单元格的统一可解释证据追踪（含决策、阈值、各项证据来源与贡献度） |
| `<timestamp>_<dataset>_explanation_audit.jsonl` | 被主动查询标注单元格的完整审计信息 |
| `<timestamp>_<dataset>_repair_candidates.csv` | 规范化等模块生成的候选修复建议 |
| `<timestamp>_<dataset>_corrected_value_standardization.csv` | 开启 `--enable_corrected_value_standardization` 时输出的格式标准化候选 |
| `<dataset>_raw_rule_proposals.json` | LLM 生成的原始列规则提案 |
| `<dataset>_raw_relationship_proposals.json` | LLM 生成的原始跨列关系提案 |
| `<dataset>_validated_relationship_rules.json` | 经数据验证保留下来的跨列关系规则 |
| `<dataset>_adaptive_mechanism_discovery.json` | 聚类发现的候选错误机制报告 |
| `<dataset>_adaptive_mechanism_validation.json` | 经主动标签验证后的机制启用/拒绝报告 |
| `<timestamp>_<dataset>_cleanem.log` | 完整运行日志 |
| `llm_usage.jsonl` | 每次 LLM API 调用的 Token 统计日志 |

---

## 数据准备与离线评测脚本

所有辅助脚本均位于 `scripts/` 目录下：

### 1. 准备 TableEG 对齐数据

若从原始 TableEG 数据集重新生成对齐表（仓库中已在 `data/tableeg_experiment_b/` 附带一份已对齐数据）：

```bash
python scripts/prepare_tableeg_experiment.py \
  --downloads /path/to/raw_tableeg \
  --output data/tableeg_experiment_b
```

### 2. 评测解释追踪的反事实忠实度（Faithfulness）

基于 `<timestamp>_<dataset>_explanation_traces.jsonl` 计算移除/仅保留核心证据后的概率变化与决策翻转率：

```bash
python scripts/evaluate_explanation_faithfulness.py \
  --trace beers=results/clean_em_beers/<timestamp>_beers_explanation_traces.jsonl \
  --output_dir results/eval_faithfulness
```

支持传入多个 `--trace NAME=PATH` 同时汇总；结果输出至 `<output_dir>/faithfulness_summary.json` 与 `.csv`。

### 3. 评测 TableEG 检测、解释语义、修复与约束对齐

```bash
python scripts/evaluate_tableeg_semantic_correctness.py \
  --dataset beers=results/clean_em_beers/<timestamp>_beers_explanation_traces.jsonl:data/tableeg_experiment_b/beers \
  --output_dir results/eval_tableeg
```

可通过 `--exclude_columns NAME=COL1,COL2` 排除指定列；结果输出至 `<output_dir>/experiment_b_metrics.json` 与 `.csv`。

---

## 运行单元测试

所有单元测试位于 `test/` 目录，项目根目录已配置 `pytest.ini` 与 `test/conftest.py`，无需手动设置 `PYTHONPATH`：

```bash
# 运行全部单元测试
pytest

# 或运行单个测试文件
pytest test/test_adaptive_mechanism_discovery.py

# 使用 unittest 运行（针对 unittest.TestCase 用例）
PYTHONPATH=. python -m unittest discover -s test
```
