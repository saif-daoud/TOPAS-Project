# Extraction

This package extracts domain components from textbooks for each role (system/user).

## Entrypoint

- Class: `topa.extraction.ExtractionPhase`
- CLI (typical):

```bash
python run_topa.py --config topa/config/config_cbt.yaml --skip-refinement --skip-annotation
```

## Inputs

- `paths.textbooks`:
  - PDF file or a folder of PDFs
  - Optional role split folders inside the textbooks path:
    - `system/`, `user/`
    - or `system_books/`, `user_books/`
- `paths.data`: dataset JSON (used by metrics)
- `extraction_params`:
  - `method`: extractor name
  - `role`: `system`, `user`, or both
  - `role_components`: components to keep per role

## Outputs

Extraction writes under:

```
<paths.output>/<domain_adj>/<ExtractorName>/extracted_components/<role>/<book>/components/*.json
```

Notes:
- Some extractors omit the `<book>` level and write directly under `<role>`.
- Usage logs are saved alongside components as `usage.json`.
- Metrics (if enabled) are written under:

```
<paths.output>/<domain_adj>/metrics/
```

## Experiment isolation

Different extraction methods already write to different folders by method name.
If you want an isolated run for the same method, set a unique output path or run in a separate output root.
