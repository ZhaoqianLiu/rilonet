# Inputs and outputs

## Metadata table

The metadata table has one row per assay profile. Source column names are
arbitrary and are mapped in YAML under `metadata.columns`.

Required logical fields are:

| Logical field | Meaning |
| --- | --- |
| `sample_id` | Profile identifier matching an omics matrix row/column |
| `participant_id` | Participant identifier used for longitudinal clustering |
| `omic` | Assay label mapped to `MG`, `MT`, or `MB` |
| `group` | Binary study group |
| `study_time` | Numeric longitudinal time |

`collection_date` and `canonical_biosample_id` are optional but recommended.
Every covariate named under `covariates.categorical` or `covariates.numeric`
must also occur in the metadata table. Exactly two canonical group levels are
currently supported.

The profile ID and omics label together must identify the correct metadata row.
Canonical-ID fallback is disabled by default because an ambiguous fallback can
silently link the wrong assay to a participant.

## Omics matrices

One matrix is supplied for each of MG, MT, and MB. The first column contains
profile IDs. The default orientation is profiles by features:

```text
sample_id  feature_1  feature_2  feature_3
S001_MG    10         0          25
S002_MG    12         3          18
```

Set `orientation: features_by_profiles` when features are rows. Profile and
feature identifiers must be unique. Values are parsed as numeric; unparseable
values become missing.

Preprocessing modes are:

- `raw`: prevalence filtering plus CLR for MG/MT, or PQN-log2 for MB.
- `transformed`: use an already transformed numeric matrix, with optional
  prevalence/IQR filtering from the YAML configuration.

The exact transformations and thresholds used are written to preprocessing QC.

## Analysis output layout

The configured `paths.output_dir` contains:

```text
analysis/
├── 01_prepared/
│   ├── matrices/{MG,MT,MB}.parquet
│   ├── sample_manifest.tsv
│   └── preparation_qc.json
├── 02_residuals/
│   ├── {MG,MT,MB}_residuals.parquet
│   ├── *_fit_diagnostics.tsv
│   └── *_design.json
├── 03_pairs/
│   ├── near/*.tsv
│   ├── lagged/*.tsv
│   └── pairing_qc.json
├── 04_candidates/
│   ├── near_*.tsv
│   ├── lagged_*.tsv
│   ├── participant_split.tsv        # only when sample splitting is enabled
│   └── screening_qc.json
├── 05_networks/
│   ├── *_edges.tsv
│   ├── *_fit_diagnostics.tsv
│   └── network_modeling_qc.json
└── 06_modules/
```

With `modeling.baseline_mode: adjust` or `change`, the corresponding folders
are named `05_networks_baseline_adjust` or `05_networks_baseline_change` and
the interpretation folders follow the same suffix.

Passing `--max-features` or `--max-edges` limits the corresponding processing
step and writes its output to a smoke-test directory.

## Multiple-testing columns

Every modeled block contains `contrast_q_block` and the active adjusted column
`contrast_q`. With `modeling.multiple_testing_scope: global`, RILoNet also
writes `contrast_q_global` and points `contrast_q` to that network-wide value.
Interpretation uses `contrast_q` by default; another available adjusted column
can be selected with `--q-column`.

## Included IBDMDB example

The ready-to-run IBDMDB inputs are together under `ibd_data/`. The metadata is
limited to the 11 columns used by `configs/ibdmdb_run.yaml`; the same folder
contains the transformed MG, filtered MT, and transformed MB matrices.
