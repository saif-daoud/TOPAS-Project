# Bundled TOPA source

This directory contains a source snapshot of the TOPA package in `topa/` so
TOPAS Studio can run independently of another repository checkout.

- Snapshot source: the workspace `TOPA-main/topa` package
- Snapshot date: 2026-09-13
- Runtime package: `website/topa`
- Website integration: late-fusion extraction and conversation-state refinement

The extraction and refinement implementations and their prompts were copied
without modification. Website-specific Azure routing, progress tracking, and
profile storage remain in `backend/`.
