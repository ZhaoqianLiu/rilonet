# Reproducibility notes

## Versioned inputs

For every reported analysis, preserve:

- the exact YAML configuration;
- the package commit hash and Python version;
- immutable input checksums or governed data-release identifiers;
- the generated QC JSON files;
- database release information and permitted annotation caches;
- random seeds and process-count settings.

RiloNet derives deterministic simulation and validation random streams from the
configured master seed. Validation draws are designed to remain the same when
the worker count changes. Set `compute.blas_threads: 1` during process-level
parallelism to avoid CPU oversubscription.

## Included study reproduction

Only one direct study-reproduction configuration is included:
`configs/simulation/end_to_end_split30_70.yaml`. It specifies:

- 200 Monte Carlo replicates;
- 240 participants and 18 visits;
- MG, MT, and MB dimensions of 12 each;
- global-null and differential truth modes;
- participant-level sample splitting with 30% for screening and 70% for
  inference;
- five candidates per network block;
- lag baseline adjustment;
- global Benjamini-Hochberg correction at 0.05.

Run it from the repository root:

```bash
rilonet simulate-study \
  --study-config configs/simulation/end_to_end_split30_70.yaml
```

The archived formal run reported near planted-edge recall of 0.78. Overall
planted-edge recall was 0.714 and lagged recall was 0.639. Under the matched
global-null simulations, the probability of any call was 0.07 overall.

## Meaning of non-planted calls

Under the global-null generator, every network call is false and the fraction
of non-planted calls is an empirical FDR quantity. Under differential truth,
compositional closure and indirect paths can induce real marginal associations
that are not direct planted structural edges. In that setting, report planted
direct-edge recovery and the non-planted call fraction, but do not automatically
call the latter strict FDR.

## External services

Interpretation contacts KEGG, NCBI, and optionally uses a user-supplied HMDB XML
export. Network availability and database content are outside the package's
control. Populate and archive caches for a reproducible interpretation run, and
respect each database's access and redistribution terms.
