# Refinement

This package refines extracted components, mainly:
- conversation states (canonicalization)
- action space refinement (micro actions)

## Entrypoint

- Class: `topa.refinement.RefinementPhase`
- CLI (typical):

```bash
python run_topa.py --config topa/config/config_cbt.yaml --skip-extraction --skip-annotation
```

## Inputs

- Extracted components path (from extraction)
- Refines files under `<...>/extracted_components/.../components/`

## Outputs

Refined components are written under:

```
<paths.output>/<domain_adj>/<ExtractorName>/refined_components/
```

Additional outputs (if enabled):
- `refinement_reports/` under `<paths.output>/<domain_adj>/`
- `annotations_micro_actions_refined.csv` saved inside the annotations folder during action-space refinement

## Experiment isolation

Refinement outputs are stored under the extraction method folder. To avoid overwrite, use a new extraction run (different output root or method folder).
