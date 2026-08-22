# Interpretation workflow

`rilonet interpret-modules` consumes completed network fits and the prepared
feature universe. It produces a reproducible, cached interpretation layer in
four steps.

## 1. Edge master and classification

For each edge, the command joins the reference- and condition-group model rows,
retains the two group effects and differential contrast, and selects discovery
edges by the requested adjusted-q column and threshold. Near edges whose two
orientations disagree in sign are excluded from the primary discovery network.

Selected edges are classified as weakened/lost, condition-emergent,
sign-reversed, strengthened, or stable according to the two group effects and
the configured near-zero threshold. The label `condition_emergent` always
refers to the group supplied with `--condition-group`.

## 2. Feature annotation

- MG features are searched in NCBI Taxonomy and returned with taxid and lineage.
- MT EC features are mapped to KEGG enzyme, pathway, and module terms.
- MB features are normalized to HMDB accessions. Full chemical taxonomy and
  pathway annotations require an official local HMDB XML export supplied with
  `--hmdb-xml`.

HTTP responses are cached by request URL under `database_cache/`. The cache
supports exact reruns and protects external services from repeated requests.
Supply `--ncbi-email` in accordance with NCBI guidance. `--offline` raises an
error on a missing cache entry; it does not silently produce partial mappings.

Database annotations can change. Archive the cache, QC JSON, and HMDB release
identifier used for any manuscript result where the applicable database terms
allow this.

## 3. Consensus Leiden modules

Near and lagged networks are partitioned separately. Edge weights are the
absolute differential contrasts. Leiden is run repeatedly at 0.75, 1.0, and
1.25 times the requested center resolution. Runs at the center resolution form
a co-assignment matrix, and stable co-assignments are partitioned once more to
obtain consensus modules.

A module is retained only if it satisfies all configured criteria:

- at least `--min-module-size` nodes;
- at least two omics layers;
- mean within-module co-assignment at least `--min-module-stability`.

The seed, every run seed, resolution, modularity, module count, node stability,
and module stability are written to output. These diagnostics should accompany
reported modules.

## 4. Enrichment

For each module and omics-specific annotation field, a one-sided
hypergeometric test compares module annotations with the complete prepared
feature universe. Benjamini-Hochberg correction is applied within each
analysis-type, omics, and annotation-field family.

Enrichment is descriptive when annotation coverage is sparse or mapping
confidence is low. Always review `annotations/annotation_qc.tsv` alongside
`module_enrichment.tsv`.

## Main outputs

| File | Contents |
| --- | --- |
| `edge_master.tsv` | One row per fitted cross-omics edge |
| `edge_classification.tsv` | Significant eligible discovery edges and classes |
| `annotations/feature_annotations.tsv` | Combined MG/MT/MB annotations |
| `annotations/annotation_qc.tsv` | Mapping coverage by omics and status |
| `module_membership.tsv` | Node-to-module assignments and stability |
| `edge_module_assignment.tsv` | Source/target module for each discovery edge |
| `module_detection_runs.tsv` | Leiden and consensus diagnostics |
| `module_summary.tsv` | Module composition and effect summaries |
| `module_enrichment.tsv` | Term enrichment statistics |
| `module_interpretation_qc.json` | Parameters, sources, mapping, and exclusion QC |
