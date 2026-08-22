# RILoNet

RILoNet is a Python framework for phenotype-associated cross-omics network signatures analysis from
longitudinal metagenomics (MG), metatranscriptomics (MT), and metabolomics (MB)
data.

It provides a workflow for:

```text
preprocessing → residualization → temporal matching → candidate screening
→ differential edge modeling → biological interpretation
```


## Requirements

- Python 3.10 or newer
- One metadata table
- Three omics matrices: MG, MT, and MB
- Internet access for NCBI and KEGG annotation
- An official HMDB XML file for full metabolite annotation and enrichment;
  without it, only HMDB accession normalization is performed

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

### 3. Run biological interpretation

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
consensus Leiden module discovery, and module enrichment. The HMDB XML is
required to reproduce full metabolite annotation and the corresponding
enrichment; omit it only when HMDB accession normalization is sufficient. See
[`docs/INTERPRETATION.md`](docs/INTERPRETATION.md).

## IBDMDB configuration

[`configs/ibdmdb_run.yaml`](configs/ibdmdb_run.yaml)
contains the settings used for the IBDMDB analysis, including filtered MT
features, lag baseline adjustment, and global multiple-testing correction.

The ready-to-run input files are together in [`ibd_data/`](ibd_data/):

```text
metadata.csv
MG_preprocessed.tsv
MT_preprocessed_filtered.tsv
MB_preprocessed.tsv
```

The metadata contains only the 11 columns required by this configuration. Run
the complete IBDMDB analysis with:

```bash
rilonet --config configs/ibdmdb_run.yaml run-core --workers 8
```

Results are written to `analysis_ibdmdb/`. Because this configuration uses
baseline adjustment, modeled networks are in
`analysis_ibdmdb/05_networks_baseline_adjust/` and can be passed directly to
the interpretation command above.

## Run the end-to-end simulation

```bash
rilonet simulate-study \
  --study-config configs/simulation/end_to_end_split30_70.yaml
```

30% of participants used for candidate screening, and 70% used for inference. Results are written
to `results/simulations/end_to_end_split30_70/`.
