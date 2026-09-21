import os
import os.path as osp
from tqdm import tqdm
import json
import glob
from typing import Any, Dict, List
from pathlib import Path

from .base_extractor import BaseExtractor

from ..utils import build_logger
logger = build_logger()

class TOPAPerBookExtractor(BaseExtractor):
    def identify_prompt(self, role: str, chapter_text: str) -> str:
        if self.mode == "task":
            prompt = f"You are an expert assistant analyzing a {self.domain} textbook chapter.\n"
            prompt += "Your goal is to extract information for building a hierarchical MDP for mathematical problem solving/theorem proving.\n"
        else:
            prompt = f"You are an expert assistant analyzing a {self.domain} textbook."
            if role == "system":
                prompt += "\nGiven a chapter, your goal is to extract dimensions of information relevant to intervention modeling.\n"
            else:
                prompt += f"\nGiven a chapter, your goal is to extract dimensions of information relevant to {self.user} modeling in {self.domain} interventions.\n"

        for idx, component in enumerate(self.components[role]):
            prompt += f"""
            {idx}. {component}:
            - Summarize the {self.SPEC[component]["definition"]}.
            - If nothing related is mentioned in the current chapter, return "None".
            """
        prompt += f"""
        ### Input Chapter Source Text:
        {chapter_text}

        ### Output Format
        Return your answer as a **plain readable string** with the following structure:
        """
        for component in self.components[role]:
            prompt += f" {component}: summary: ... \n "
        return prompt

    def string_to_json_prompt(self, role: str, extracted_text: str) -> str:
        prompt = f"""
        You are an expert assistant.

        You are given the following **extracted text** from a {self.domain} textbook chapter:
        ---
        {extracted_text}
        ---
        Transform this text into the following JSON structure, **without changing the content**:"""

        prompt += "{{ \n"
        for component in self.components[role]:
            prompt += f""" "{component}": {{ "summary": "..." }}, \n"""
        prompt += "}\n"
        prompt += """
        - Each "summary" should contain exactly what is described in the extracted text for that dimension.
        - If a dimension was marked "None" in the text, keep it as "None".
        - Make sure the JSON is valid and properly formatted.
"""
        return prompt.strip()

    def extract_component_prompt(self, input_content: str, component: str, relevant_component=None) -> str:
        if component =="micro_actions":
            prompt = f"""
            You are an assistant that extracts {component} from provided summaries of {self.domain} textbooks.
            ### Input: 
            1. A list of **macro actions** extracted from a {self.domain} textbook.
            2. A summary of the source chapters.

            ### Task: 
            For each macro action extract the corresponding micro actions 

            ### Definition:
            {component} are {self.SPEC[component]["definition"]}

            ### Description:
            {self.SPEC[component]["description"]}
        
            ### Source Text:
            {input_content}

            ### Input Macro Actions:
            {relevant_component}

            ### Return the result as a **valid JSON object**, strictly following this format:  
            {self.SPEC[component]["format"]}

            ### Output:
            """
        else:
            prompt = f"""
            You are an assistant that extracts {component} from provided summaries of {self.domain} textbooks.
            
            ### Definition:
            {component} are {self.SPEC[component]["definition"]}

            {"### Description:" if self.SPEC[component]["description"] != "" else ""}
            {self.SPEC[component]["description"]}
        
            ### Source Text:
            {input_content}

            ### Return the result as a **valid JSON object**, strictly following this format:  
            {self.SPEC[component]["format"]}

            ### Output:
            """
        return prompt

    def summarize_book(self, role: str, save_dir: str, book_index: int, book_path: str) -> str:
        toc, chapter_texts = self.extract_chapters_and_toc(book_path)
        all_summaries: List[Dict[str, Any]] = []

        for i, (_, content) in tqdm(enumerate(chapter_texts), total=len(chapter_texts)):
            prompt = self.identify_prompt(role, content)
            output, _, _ = self.run_llm_query(prompt)

            conversion_prompt = self.string_to_json_prompt(role, output)
            output2, _, _ = self.run_llm_query(conversion_prompt)

            try:
                parsed = self.parse_json(output2)
            except:
                parsed = self.try_again(conversion_prompt)

            all_summaries.append(parsed)

        s_path_book = osp.join(save_dir, f"summaries_book_{book_index}.json")
        with open(s_path_book, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)

        return s_path_book

    def _resolve_single_pdf(self, pdf_path: str | None = None) -> Path:
        if pdf_path:
            p = Path(pdf_path).expanduser()
        else:
            p = Path(self.textbooks_path).expanduser()

        if p.is_file() and p.suffix.lower() == ".pdf":
            return p

        if p.is_dir():
            pdfs = sorted(p.glob("*.pdf"))
            if len(pdfs) == 0:
                raise FileNotFoundError(f"No PDF files found in directory: {p}")
            if len(pdfs) > 1:
                raise RuntimeError(
                    f"TOPAPerBookExtractor supports only ONE book, but found {len(pdfs)} PDFs in: {p}\n"
                    f"Either set paths.textbooks to a single .pdf file, or pass extraction_params.pdf_path.\n"
                    f"Found:\n" + "\n".join([f"- {x}" for x in pdfs])
                )
            return pdfs[0]

        raise FileNotFoundError(f"Invalid textbooks_path/pdf_path: {p}")

    def extract_per_book(self, role: str, save_dir: str, s_path_book: str) -> str:
        with open(s_path_book, "r", encoding="utf-8") as f:
            book_summary = json.load(f)

        os.makedirs(save_dir, exist_ok=True)

        relevant_component = []
        for component in self.components[role]:
            component_summary = ""
            for chp_idx, summary in enumerate(book_summary):
                component_summary += f"chapter_index_{chp_idx} {summary[component]['summary']}"

            prompt = self.extract_component_prompt(component_summary, component, relevant_component)
            output, _, _ = self.run_llm_query(prompt)

            try:
                parsed = self.parse_json(output)
            except:
                parsed = self.try_again(prompt)

            with open(osp.join(save_dir, f"{component}.json"), "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)

            if component == "macro_actions":
                relevant_component = parsed

        return save_dir

    def extract(self, *, role: str, **kwargs) -> str:
        pdf = self._resolve_single_pdf(self.textbooks_path)
        book_name = pdf.stem

        save_dir = osp.join(str(self.output_path), self.domain, "TOPAPerBookExtractor", "extracted_components", role, book_name)
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"[TOPA] Extracting components from book: {book_name}...")
        s_path = self.summarize_book(role, save_dir, book_index=0, book_path=str(pdf))

        book_out_dir = osp.join(save_dir, "components")
        self.extract_per_book(role, book_out_dir, s_path)

        return book_out_dir

class TOPAOurExtractor(TOPAPerBookExtractor):
    def fusion_prompt(self, component, extracted_info):
        prompt = f"""
        You will be given multiple sources (textbooks) with designs for {self.domain}.

        Your job is to MERGE overlapping/duplicate items across sources into a SINGLE, de-duplicated structure.

        ### CRITICAL OUTPUT RULES:
        - Output MUST be valid JSON only (UTF-8), with no leading/trailing text, no Markdown, no comments.
        - Do not invent fields not specified in the required schema.
        - Use double quotes for all keys and string values.
        - No trailing commas.
        - Use concise, plain-English phrasing.

        ### TASK: 
        Merge the {"action spaces" if component == "micro_actions" else component} designed for {self.domain} from different books.

        ### INPUT:
        {extracted_info}

        OUTPUT SCHEMA (JSON only):
        {self.SPEC[component]["format"]}        
        """
        return prompt

    def extract(self, *, role: str, fusion_type: str, **kwargs):
        # Put outputs inside outputs/<domain>/...
        save_dir = osp.join(str(self.output_path), self.domain, f"TOPAOurExtractor_{fusion_type}Fusion", "extracted_components", role)
        os.makedirs(save_dir, exist_ok=True)

        final_output_dir = osp.join(save_dir, "merged_components")
        os.makedirs(final_output_dir, exist_ok=True)

        pdf_files = kwargs.get("pdf_files", None)
        if pdf_files is None:
            pdf_files = sorted(glob.glob(osp.join(str(self.textbooks_path), "*.pdf")))
        else:
            pdf_files = list(pdf_files)
        book_names = [osp.splitext(osp.basename(p))[0] for p in pdf_files]

        ### Extraction:
        for book_idx, book_path in enumerate(pdf_files):
            logger.info(f"[TOPA] Creating summaries for book: {book_names[book_idx]}...")
            self.summarize_book(role, save_dir, book_idx, book_path)

        if fusion_type == "Late":
            ###  Late fusion:
            logger.info("[TOPA] Extracting components per book...")
            for book_idx, _ in tqdm(enumerate(pdf_files), total=len(pdf_files)):
                summary_path = osp.join(save_dir, f"summaries_book_{book_idx}.json")
                self.extract_per_book(role, osp.join(save_dir, f"book_{book_idx}"), summary_path)

            logger.info("[TOPA] Fusing extracted components...")
            for component in tqdm(self.components[role], total=len(self.components[role])):
                all_books_component = ""
                for book_idx, _ in enumerate(pdf_files):
                    all_books_component += f"\nBook index {book_idx}:\n"
                    with open(osp.join(save_dir, f"book_{book_idx}", f"{component}.json"), "r", encoding="utf-8") as f:
                        all_books_component += f.read()

                prompt = self.fusion_prompt(component, all_books_component)
                output, _, _ = self.run_llm_query(prompt)

                try:
                    parsed = self.parse_json(output)
                except:
                    parsed = self.try_again(prompt)

                with open(osp.join(final_output_dir, f"{component}.json"), "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)

        elif fusion_type == "Early":
            ###  Early fusion:
            logger.info("[TOPA] Fusing summaries...")
            all_book_summaries = []
            for book_idx, _ in tqdm(enumerate(pdf_files), total=len(pdf_files)):
                with open(osp.join(save_dir, f"summaries_book_{book_idx}.json"), "r", encoding="utf-8") as f:
                    all_book_summaries.append(json.load(f))

            logger.info("[TOPA] Extracting components from fused summaries...")
            relevant_component = []
            for component in tqdm(self.components[role], total=len(self.components[role])):
                component_summary = ""
                for book_idx, summaries in enumerate(all_book_summaries):
                    component_summary += f"\nBook_name_{book_names[book_idx]}\n"
                    for chp_idx, summary in enumerate(summaries):
                        component_summary += f"chapter_index_{chp_idx} {summary[component]['summary']}\n"

                prompt = self.extract_component_prompt(component_summary, component, relevant_component)
                output, _, _ = self.run_llm_query(prompt)

                try:
                    parsed = self.parse_json(output)
                except:
                    parsed = self.try_again(prompt)

                if component == "macro_actions":
                    relevant_component = parsed

                with open(osp.join(final_output_dir, f"{component}.json"), "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        return final_output_dir