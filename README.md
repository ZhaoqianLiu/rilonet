# RILoNet

RILoNet is a Python framework for phenotype-associated cross-omics network signatures analysis from
longitudinal metagenomics (MG), metatranscriptomics (MT), and metabolomics (MB)
data.

It provides one workflow for:

```text
preprocessing → residualization → temporal matching → candidate screening
→ differential edge modeling → validation → biological interpretation
```


## Requirements

- Python 3.10 or newer
- One metadata table
- Three omics matrices: MG, MT, and MB
- Internet access for NCBI and KEGG annotation
- Optional official HMDB XML file for full metabolite annotation

Input column names, group labels, covariates, time units, matching windows, and
model settings are defined in YAML. See
[`configs/analysis_example.yaml`](configs/analysis_example.yaml) and
[`docs/INPUTS_AND_OUTPUTS.md`](docs/INPUTS_AND_OUTPUTS.md).

## Installation

```bash
git clone https://github.com/ZhaoqianLiu/rilonet.git
cd rilonet

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[interpretation]"
```

Confirm that the command is available:

```bash
rilonet --help
```

## Run RILoNet

### 1. Configure the input

Copy the example configuration and update the metadata, matrix, group, and
covariate settings:

```bash
cp configs/analysis_example.yaml configs/my_analysis.yaml
```

### 2. Run the main framework

Run preprocessing through network modeling in one command:

```bash
rilonet --config configs/my_analysis.yaml run-core --workers 8
```

The same workflow can be run stage by stage:

```bash
rilonet --config configs/my_analysis.yaml prepare
rilonet --config configs/my_analysis.yaml residualize --omic all --workers 8
rilonet --config configs/my_analysis.yaml pair
rilonet --config configs/my_analysis.yaml screen
rilonet --config configs/my_analysis.yaml model --workers 8
```

Results are written to the `paths.output_dir` specified in the YAML file.

### 3. Validate selected networks

```bash
rilonet --config configs/my_analysis.yaml validate \
  analysis/05_networks/near_MG_MB_edges.tsv \
  --workers 8
```

### 4. Run biological interpretation

```bash
rilonet interpret-modules \
  --analysis-dir analysis \
  --network-dir analysis/05_networks \
  --reference-group reference \
  --condition-group condition \
  --q-column contrast_q \
  --q-threshold 0.05 \
  --ncbi-email you@example.org \
  --hmdb-xml /path/to/hmdb_metabolites.xml
```

Interpretation includes feature annotation, differential-edge classification,
consensus Leiden module discovery, and module enrichment. The HMDB argument can
be omitted when only HMDB accession normalization is required. See
[`docs/INTERPRETATION.md`](docs/INTERPRETATION.md).

## IBDMDB configuration

[`configs/ibdmdb_filtered_baseline_adjust.yaml`](configs/ibdmdb_filtered_baseline_adjust.yaml)
contains the settings used for the IBDMDB analysis, including filtered MT
features, lag baseline adjustment, and global multiple-testing correction.


## Reproduce the end-to-end simulation

```bash
rilonet simulate-study \
  --study-config configs/simulation/end_to_end_split30_70.yaml
```

30% of participants used for candidate screening, and 70% used for inference. Results are written
to `results/simulations/end_to_end_split30_70/`.

## Run the tests

```bash
python -m pip install -e ".[interpretation,dev]"
python -m pytest
```
