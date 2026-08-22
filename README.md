# RILoNet

RiloNet infers differential cross-omics network signatures from longitudinal
metagenomics (MG), metatranscriptomics (MT), and metabolomics (MB) data. This
repository contains the runnable core framework, the simulation program, and
the feature-annotation and consensus-module interpretation workflow.

The release is configuration driven: cohort-specific column names, group
labels, covariates, time units, preprocessing, matching windows, model choices,
and output directories are defined in YAML rather than hard-coded in the
package.

## What to upload to GitHub

Upload the **contents of this `rilonet_github/` directory as the repository
root**. Do not upload the parent `multi-omics/` or the old `rilonet/` directory.
After upload, the top level of GitHub should look like this:

```text
README.md
README.zh-CN.md
pyproject.toml
requirements.txt
src/rilonet/                  # main RiloNet, simulation, interpretation code
configs/                      # analysis configs and one reproduction config
tests/
docs/
```

The actual Python code to publish is under `src/rilonet/`:

- `prepare.py`, `residualize.py`, `matching.py`, `screening.py`, and
  `modeling.py`: main RiloNet framework;
- `simulate.py` and `simulation_study.py`: simulation code;
- `interpretation.py`: annotation, consensus Leiden modules, and enrichment;
- `cli.py`: all command-line entry points.


## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[interpretation]"
rilonet --help
```

The `interpretation` extra installs `igraph` and `leidenalg`, which are needed
for consensus Leiden module discovery. Core network inference and simulation
can instead be installed with `python -m pip install -e .`.

## Optional software smoke example

Generate a fully synthetic data set, including a ready-to-run configuration and
the planted truth. This checks that the software runs; it is not the reported
simulation reproduction.

```bash
rilonet simulate \
  --output-dir examples/synthetic \
  --subjects 60 \
  --visits 10 \
  --seed 20260715
```

Run preprocessing through network modeling:

```bash
rilonet --config examples/synthetic/config.yaml run-core --workers 4
```

The generated config uses block-wise multiple-testing correction. For a single
network-wide Benjamini-Hochberg family, set
`modeling.multiple_testing_scope: global` before running `model`.

A shorter integration check is available, but its limited output must not be
interpreted scientifically:

```bash
rilonet --config examples/synthetic/config.yaml run-core \
  --max-features 5 --max-edges 2
```

## Real-data workflow

Copy `configs/analysis_example.yaml`, point it to an assay-level metadata table
and three omics matrices, and adapt the column mappings and group labels. The
standard table orientation is profiles by features; features by profiles is
also supported through the YAML setting.

`configs/ibdmdb_filtered_baseline_adjust.yaml` preserves the main IBDMDB
analysis settings: nonIBD versus pooled CD/UC, transformed MG/MT/MB inputs,
filtered MT features, lag baseline adjustment, and global contrast correction.
Its governed IBDMDB metadata and matrices are not distributed here; the config
documents the expected paths and source columns.

Run each auditable stage separately:

```bash
rilonet --config configs/my_analysis.yaml prepare
rilonet --config configs/my_analysis.yaml residualize --omic all --workers 8
rilonet --config configs/my_analysis.yaml pair
rilonet --config configs/my_analysis.yaml screen
rilonet --config configs/my_analysis.yaml model --workers 8
```

For the supplied IBDMDB configuration, the corresponding commands are:

```bash
rilonet --config configs/ibdmdb_filtered_baseline_adjust.yaml run-core \
  --workers 8
rilonet interpret-modules \
  --analysis-dir analysis_ibdmdb_filtered_baseline_adjust \
  --network-dir analysis_ibdmdb_filtered_baseline_adjust/05_networks_baseline_adjust \
  --output-dir analysis_ibdmdb_filtered_baseline_adjust/06_modules_baseline_adjust \
  --reference-group nonIBD \
  --condition-group IBD \
  --q-column contrast_q \
  --hmdb-xml /path/to/hmdb_metabolites.xml
```

The stages write prepared matrices and QC, screening residuals, matched assay
pairs, screened candidates, and fitted near/lagged networks under the configured
analysis directory. See [docs/INPUTS_AND_OUTPUTS.md](docs/INPUTS_AND_OUTPUTS.md)
for the required schema and output layout.

Candidate screening uses covariate-adjusted residuals. Final inference returns
to transformed observed measurements and decomposes time-varying predictors
into within- and between-participant components. Near-contemporaneous edges are
fit in both orientations; lagged edges retain direction and can optionally use
explicit baseline adjustment or change models.

For confirmatory sample splitting, enable
`screening.sample_split.enabled: true`. Screening then uses discovery
participants and modeling/validation uses held-out inference participants. A
full-data reuse analysis remains discovery-stage evidence even if its threshold
was calibrated by simulation.

## Validation

Validation selects edges using the configured differential-contrast q-value and
performs participant-level bootstrap and permutation resampling:

```bash
rilonet --config configs/my_analysis.yaml validate \
  analysis/05_networks/near_MG_MB_edges.tsv \
  --workers 8
```

Group labels are permuted at participant level within configured strata.
Lagged edges additionally receive a within-participant circular-shift temporal
permutation.

## Interpretation

Interpretation annotates the full prepared feature universe, classifies
significant differential edges, discovers stable consensus Leiden modules, and
tests within-module annotation enrichment:

```bash
rilonet interpret-modules \
  --analysis-dir analysis \
  --reference-group reference \
  --condition-group condition \
  --q-column contrast_q \
  --q-threshold 0.05 \
  --ncbi-email you@example.org
```

MG taxonomy is queried from NCBI, MT enzyme/pathway mappings from KEGG, and MB
annotations can be read from an official local HMDB XML export using
`--hmdb-xml`. Responses are cached under the interpretation output directory.
Use `--offline` only after the required cache has been populated. Database
versions and mapping QC are recorded in `module_interpretation_qc.json`.

If a non-default network directory is being interpreted, for example a
baseline-adjusted fit, supply both paths explicitly:

```bash
rilonet interpret-modules \
  --analysis-dir analysis \
  --network-dir analysis/05_networks_baseline_adjust \
  --output-dir analysis/06_modules_baseline_adjust \
  --reference-group nonIBD \
  --condition-group IBD \
  --hmdb-xml /path/to/hmdb_metabolites.xml
```

See [docs/INTERPRETATION.md](docs/INTERPRETATION.md) for database and consensus
module details.

## Simulation studies

The repository provides exactly one study-reproduction configuration: the
end-to-end formal validation using 30% of participants for screening and 70%
for inference.

```bash
rilonet simulate-study \
  --study-config configs/simulation/end_to_end_split30_70.yaml
```

This is the 200-replicate, `n=240`, 18-visit, baseline-adjusted, global-BH
configuration that produced near planted-edge recall of **0.78**. It also runs
the matched global-null condition. Outputs are written to
`results/simulations/end_to_end_split30_70/`.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the exact design and
interpretation of the reported metrics.

## Tests

```bash
python -m pip install -e ".[interpretation,dev]"
python -m pytest
```

## Citation and license

Citation metadata and a software license have not yet been assigned. Add the
project authors, repository URL, preferred citation, and the chosen license
before making the repository public. Without an explicit license, GitHub users
do not receive general permission to reuse or redistribute the code.
