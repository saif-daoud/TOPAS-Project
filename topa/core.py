import json
import logging
from pathlib import Path
from typing import Any, Optional

from .extraction import ExtractionPhase
from .refinement import RefinementPhase
from .annotation import AnnotationManager
from .rl import RLTrainingPhase
from .config import Config

class TOPA:
    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        self.cfg = config
        self.domain = self.cfg.get_adj()
        self.log = logger or logging.getLogger("topa")
        self.agent = None

        # Cache convention: outputs/<domain>/
        self.cache_dir = (Path(self.cfg.output_path) / self.domain)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.extracted_path = self.cache_dir / "extracted.json"
        self.refined_path = self.cache_dir / "refined.json"
        self.annotations_path = self.cache_dir / "annotations.json"
        self.rl_path = self.cache_dir / "rl.json"

    def _save_artifact(self, obj: Any, json_path: Path):
        json_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_artifact(self, json_path: Path):
        if json_path.exists():
            return json.loads(json_path.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"Missing artifact: {json_path}")

    def run_pipeline(
        self,
        run_extraction: bool = True,
        run_refinement: bool = True,
        run_annotation: bool = True,
        run_rl: bool = False
    ):

        extracted_components = None
        refined_components = None
        annotations = None

        # 1) Extraction
        if run_extraction:
            if self.extracted_path.exists():
                self.log.info("Loading cached extraction: %s", self.extracted_path.name)
                extracted_components = self._load_artifact(self.extracted_path)
            else:
                self.log.info("[TOPA] Starting extraction...")
                extraction_phase = ExtractionPhase(
                    self.cfg.domain,
                    self.cfg.domain_params,
                    self.cfg.textbooks_path,
                    self.cfg.data_path,
                    self.cfg.output_path,
                    self.cfg.api_key,
                    self.cfg.mode,
                    self.cfg.seed,
                )
                extracted_components = extraction_phase.run(extraction_params=self.cfg.extraction_params)
                self._save_artifact(extracted_components, self.extracted_path)
        else:
            self.log.info("[TOPA] Skipping extraction.")

        # 2) Refinement
        if run_refinement:
            if extracted_components is None:
                # allow refinement-only runs to load extraction if present
                if self.extracted_path.exists():
                    extracted_components = self._load_artifact(self.extracted_path)
                else:
                    raise RuntimeError("Refinement requires extracted components (run extraction or resume).")

            if self.refined_path.exists():
                self.log.info("[TOPA] Loading cached refinement: %s", self.refined_path.name)
                refined_components = self._load_artifact(self.refined_path)
            else:
                self.log.info("[TOPA] Refining extracted knowledge...")
                refinement_phase = RefinementPhase(self.domain, self.cfg.api_key)
                refined_components = refinement_phase.run(extracted_components)
                self._save_artifact(refined_components, self.refined_path)
        else:
            self.log.info("[TOPA] Skipping refinement.")

        # 3) Annotation
        if run_annotation:
            if refined_components is None:
                if self.refined_path.exists():
                    refined_components = self._load_artifact(self.refined_path)
                else:
                    raise RuntimeError("Annotation requires refined components (run refinement or resume).")

            if self.annotations_path.exists():
                self.log.info("[TOPA] Loading cached annotations: %s", self.annotations_path.name)
                annotations = self._load_artifact(self.annotations_path)
            else:
                self.log.info("[TOPA] Annotating data...")
                annotator = AnnotationManager(
                    self.cfg.domain,
                    self.cfg.domain_params,
                    self.cfg.data_path,
                    self.cfg.output_path,
                    self.cfg.annotations_params,
                    self.cfg.api_key,
                )
                annotations = annotator.annotate(refined_components)
                self._save_artifact(annotations, self.annotations_path)
        else:
            self.log.info("[TOPA] Skipping annotation.")

        # 4) RL training
        if run_rl:
            if refined_components is None:
                if self.refined_path.exists():
                    refined_components = self._load_artifact(self.refined_path)
                else:
                    raise RuntimeError("RL requires refined components (run refinement or resume).")
            if annotations is None:
                if self.annotations_path.exists():
                    annotations = self._load_artifact(self.annotations_path)
                else:
                    raise RuntimeError("RL requires annotations (run annotation or resume).")

            self.log.info("[TOPA] Starting RL training...")
            rl_phase = RLTrainingPhase(self.cfg)
            rl_out = rl_phase.run(
                extracted_components=extracted_components,
                refined_components=refined_components,
                annotations=annotations,
            )
            self._save_artifact(rl_out, self.rl_path)
            self.agent = rl_out

        self.log.info("[TOPA] Pipeline completed.")
        return self.agent

    def get_agent(self):
        if self.agent is None:
            raise RuntimeError("Agent not trained yet. Call topa.run()")
        return self.agent