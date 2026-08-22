# RILoNet 发布目录说明

这个目录是从当前最新实现整理出的独立 GitHub 版本，包含三部分：
RILoNet 纵向多组学网络主框架、simulation 程序，以及 annotation、共识
Leiden 模块识别和富集分析组成的 interpretation 流程。历史日志、缓存、
大体积结果、真实队列数据和本机环境均未复制进来。

## 上传哪个目录

只上传 `/home/lzz/multi-omics/rilonet_github/` **里面的全部内容**，把它作为
GitHub 仓库根目录。不要上传外层 `multi-omics/`，也不要上传旧的 `rilonet/`。

代码在 `src/rilonet/`：

- 主框架：`prepare.py`、`residualize.py`、`matching.py`、`screening.py`、
  `modeling.py`
- Simulation：`simulate.py`、`simulation_study.py`
- Interpretation：`interpretation.py`
- 命令行入口：`cli.py`

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[interpretation]"
```

## 最小示例

```bash
rilonet simulate --output-dir examples/synthetic --subjects 60 --visits 10
rilonet --config examples/synthetic/config.yaml run-core --workers 4
```

真实数据分析时，复制 `configs/analysis_example.yaml`，修改 metadata、三张
组学矩阵、分组、协变量和时间字段映射，然后依次运行：

```bash
rilonet --config configs/my_analysis.yaml prepare
rilonet --config configs/my_analysis.yaml residualize --omic all --workers 8
rilonet --config configs/my_analysis.yaml pair
rilonet --config configs/my_analysis.yaml screen
rilonet --config configs/my_analysis.yaml model --workers 8
```

`configs/ibdmdb_filtered_baseline_adjust.yaml` 保留了 IBDMDB 主分析参数
（nonIBD 对合并后的 CD/UC、过滤 MT、lag baseline adjust、global BH），但受
数据许可约束，真实 metadata 和组学矩阵没有放进发布目录。

Interpretation 示例：

```bash
rilonet interpret-modules \
  --analysis-dir analysis \
  --reference-group nonIBD \
  --condition-group IBD \
  --q-column contrast_q \
  --hmdb-xml /path/to/hmdb_metabolites.xml
```

论文 simulation 只保留一个可直接复现配置：30% participants 用于 screening、
70% 用于 inference 的 end-to-end formal validation（200 次重复，near recall =
0.78）：

```bash
rilonet simulate-study \
  --study-config configs/simulation/end_to_end_split30_70.yaml
```

完整参数、输入输出格式、数据库缓存和统计解释请看英文
[README](README.md) 及 `docs/`。公开仓库前仍需由项目负责人确定作者、引用信息和
软件许可证；这些信息不应由整理代码时擅自假定。
