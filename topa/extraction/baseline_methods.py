import PyPDF2
import json
import glob
import os 
import os.path as osp
import re
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from typing import Any, Dict, List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time

from .base_extractor import BaseExtractor

from ..utils import build_logger
logger = build_logger()

class LC_Full(BaseExtractor):

    def extract_prompt(self, role, text):
        prompt=f"""
        You are an expert assistant that analyzes a {self.domain} textbook and extracts components in a single pass:
        """
        if role=="system":
            prompt+=f"""
                **Action Space (Macro + Micro Actions)**: 
                    macro actions are:
                    {self.SPEC["macro_actions"]["definition"]}
                    {self.SPEC["macro_actions"]["description"]}

                    micro actions are:
                    {self.SPEC["micro_actions"]["definition"]}
                    {self.SPEC["micro_actions"]["description"]}
            """
        for component in self.components[role]:
            if component in ["macro_actions", "micro_actions"]:
                continue
            prompt+=f"""
                **{component}**:
                {self.SPEC[component]["definition"]}
                {self.SPEC[component]["description"]}
            """
        prompt+="""  
        ### Return a single string with the following structure: """
        for component in self.components[role]:
            if component in ["macro_actions"]:
                continue
            prompt+=f"""
                **{component}**:
                {self.SPEC[component]["format_str"]}
            """  
        prompt+=f"""
            **Book Source Text:**  
            {text}
        """ 
        return prompt
    
    def extract(self, *, role: str, **kwargs) -> str:
        pdf_path = Path(self.textbooks_path).expanduser()
        book_name = pdf_path.stem

        save_dir = osp.join(str(self.output_path), self.domain, self.__class__.__name__, "extracted_components", role, book_name)
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"[TOPA] Extracting components from book: {book_name}...")
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        prompt = self.extract_prompt(role, text)
        output_str, _, _ = self.run_llm_query(prompt)
        conversion_prompt = self.string_to_json_prompt(role, output_str)
        output, _, _ = self.run_llm_query(conversion_prompt)
        try:
            parsed_output = self.parse_json(output)
        except:
            parsed_output = self.try_again(conversion_prompt)

        output_path = osp.join(save_dir, "all_components.json")
        with open(output_path,"w", encoding="utf-8") as f:
            json.dump(parsed_output, f, indent=2, ensure_ascii=False)

        return output_path

class Chap_Seq(BaseExtractor):

    def extract_prompt(self, role, last_output, chapter_content, chapter_index):
        prompt=f"""
        You are an expert assistant analyzing a {self.domain} textbook to incrementally build structured knowledge. 
        You are given:
            - A single **chapter text** (current input).  
            - The **last output** (the accumulated structure from earlier chapters).  

        Your task: Update the following sections based on the new chapter:  
        """
        for idx, component in enumerate(self.components[role]):
            prompt+=f"""
            {idx}. **{component}** \n
            """
        prompt+="### Definitions \n"

        for idx, component in enumerate(self.components[role]):
            prompt+=f"""
            {component} 
            {self.SPEC[component]["definition"]}
            {self.SPEC[component]["description"]}
            """

        prompt+="""
          ### Rules:  
        - Extract **the most relevant information**.  
        - You may **add, update, or delete** entries.  
        - The output should be a **string**, not JSON. 
        """

        prompt+=f"""
        ### Inputs
        **Current Chapter (index {chapter_index}):**  
        {chapter_content}

        **Last Output:**  
        {last_output}
        """
        prompt+="""
        ### Format 
        Return a single string with this format: 
        """
        for idx, component in enumerate(self.components[role]):
            prompt+=f"""
                {self.SPEC[component]["format_str"]} \n
            """
        prompt+="### Output: "
        return prompt

    def extract(self, *, role: str, **kwargs) -> str:
        pdf_path = Path(self.textbooks_path).expanduser()
        book_name = pdf_path.stem

        save_dir = osp.join(str(self.output_path), self.domain, self.__class__.__name__, "extracted_components", role, book_name)
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"[TOPA] Extracting components from book: {book_name}...")
        toc, chapter_texts = self.extract_chapters_and_toc(str(pdf_path))

        last_output=""
        for i, (content_type, content) in tqdm(enumerate(chapter_texts), total=len(chapter_texts)):
            prompt = self.extract_prompt(role, last_output, content, i)
            output_str, _, _=self.run_llm_query(prompt)
            print(output_str)
            last_output=output_str

        conversion_prompt = self.string_to_json_prompt(role, output_str)
        output, _, _=self.run_llm_query(conversion_prompt)
        print(output)

        try:
            parsed_output = self.parse_json(output)
        except:
            parsed_output = self.try_again(conversion_prompt)

        output_path = osp.join(save_dir, "all_components.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_output, f, indent=2, ensure_ascii=False)

        return output_path
    
class Chunk_RAG(BaseExtractor):

    def get_embedding(self, text, model="text-embedding-3-small"):
        # Support either TOPA_API_KEY (preferred) or OpenAI env vars.
        api_key = os.getenv("TOPA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            raise KeyError(
                "Missing API key. Set TOPA_API_KEY (recommended) or OPENAI_API_KEY/OPENAI_KEY in your .env"
            )
        client = OpenAI(api_key=api_key)
        return client.embeddings.create(model=model, input=text).data[0].embedding

    def chunk_text(self, text: str, max_tokens: int = 3000, overlap: int = 100):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    def build_chunk_embeddings(self, chapter_texts, max_tokens=3000, overlap=100):
        chunk_embeddings = [] 
        for chapter_idx, (content_type,  chapter) in enumerate(chapter_texts):
            chunks = self.chunk_text(chapter, max_tokens=max_tokens, overlap=overlap)
            for chunk_idx, chunk in enumerate(chunks):
                emb = self.get_embedding(chunk)
                chunk_embeddings.append(((chapter_idx, chunk_idx), emb, chunk))
        return chunk_embeddings

    def retrieve_top_chunks(self, query, chunk_embeddings, top_k=3):
        query_emb = self.get_embedding(query)
        scores = []
        for (chapter_idx, chunk_idx), emb, chunk in chunk_embeddings:
            sim = cosine_similarity(np.array(query_emb).reshape(1, -1), 
                                    np.array(emb).reshape(1, -1)).item()
            scores.append(((chapter_idx, chunk_idx), sim, chunk))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

        retrieved = []
        for (chapter_idx, chunk_idx), _, chunk in scores:
            retrieved.append(f"chapter{chapter_idx}_chunk{chunk_idx}\n{chunk}")
        return retrieved
    
    def extract_component_prompt(self, input_content, component, relevant_component=None):
        if component =="micro_actions":
            prompt=f"""
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
            prompt=f"""
            You are an assistant that extracts {component} from provided summaries of {self.domain} textbooks.
            
            ### Definition:
            {component} are {self.SPEC[component]["definition"]}

            ### Description:

            {self.SPEC[component]["description"]}
        
            ### Source Text:
            {input_content}

            ### Return the result as a **valid JSON object**, strictly following this format:  
            {self.SPEC[component]["format"]}

            ### Output:
            """
        return prompt
    
    def get_queries(self, component):
        d = {
            "knowledge_graph": [
                f"Key entities and concepts in {self.domain} and how they are related for modeling long-term user knowledge",
                f"Relationships between the {self.user}, internal states, behaviors, environment, and outcomes in {self.domain}",
                f"Core concepts in {self.domain} that should be represented in a knowledge graph for the {self.system}"
            ],

            "conversation_states": [
                f"Important state variables that a {self.system} should track during each {self.interaction_unit} in {self.domain}",
                f"User-related dimensions such as goals, emotions, engagement, and progress relevant to {self.domain}",
                f"Observable and latent state features used for dialogue management and decision-making in {self.domain}"
            ],

            "macro_actions": [
                f"High-level dialogue phases and strategies used by the {self.system} in {self.domain}",
                f"Examples of macro-level interventions or dialogue strategies in {self.domain} interactions",
                f"Typical high-level goals and phases structuring a {self.domain} conversation"
            ],

            "micro_actions": [
                f"Low-level dialogue actions or utterance-level interventions used by the {self.system} in {self.domain}",
                f"Examples of micro-interventions executed at each interaction step in {self.domain} conversations",
                f"Granular dialogue moves that operationalize high-level strategies in {self.domain}"
            ],

            "cautions": [
                f"Common risks or mistakes a {self.system} should avoid during a {self.domain} {self.interaction_unit}",
                f"Unsafe, ineffective, or undesirable behaviors in {self.domain} dialogue with a {self.user}",
                f"Ethical, practical, or interactional warnings relevant to {self.domain} conversational systems"
            ],

            "rules": [
                f"Rules or constraints governing appropriate behavior of a {self.system} in {self.domain}",
                f"Guidelines that define what the {self.system} must or must not do in {self.domain} interactions",
                f"Decision-making constraints and policies for managing {self.domain} dialogue"
            ],

            "user_profile": [
                f"Stable {self.user} attributes and individual differences that shape how the {self.user} thinks, feels, behaves, and responds in {self.domain} {self.interaction_unit}s",
                f"{self.user} background/context factors (experience level, motivations, values, constraints) that influence engagement with the {self.system} in {self.domain}",
                f"User persona/profile dimensions useful for simulating realistic {self.user} responses in {self.domain} dialogues (e.g., communication style, preferences, typical patterns)"
            ],
        }

        return d[component]

    def extract(self, *, role: str, **kwargs) -> str:
        pdf_path = Path(self.textbooks_path).expanduser()
        book_name = pdf_path.stem

        save_dir = osp.join(str(self.output_path), self.domain, self.__class__.__name__, "extracted_components", role, book_name)
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"[TOPA] Extracting components from book: {book_name}...")
        toc, chapter_texts = self.extract_chapters_and_toc(str(pdf_path))
        chunk_embeddings = self.build_chunk_embeddings(chapter_texts) 

        retrieval_log = {
            "method": "Chunk_RAG",
            "role": role,
            "book_name": book_name,
            "components": {}
        }

        relevant_component=[]
        for component in tqdm(self.components[role], total=len(self.components[role])):

            retrieved_chunks = []
            for query in self.get_queries(component):
                retrieved_chunks.extend(self.retrieve_top_chunks(query, chunk_embeddings, top_k=10))
            retrieved_chunks = list(dict.fromkeys(retrieved_chunks))

            comp_log = {"retrieved": [], "chapter_counts": {}}
            for s in retrieved_chunks:
                pref = s.split('\n', 1)[0]
                mm = re.match(r"chapter(\d+)_chunk(\d+)", pref)
                if mm:
                    ch = int(mm.group(1))
                    ck = int(mm.group(2))
                    key = f"{0}:{ch}"
                    comp_log["retrieved"].append({"book": 0, "chapter": ch, "chunk": ck})
                    comp_log["chapter_counts"][key] = comp_log["chapter_counts"].get(key, 0) + 1
            retrieval_log["components"][component] = comp_log

            prompt = self.extract_component_prompt("\n".join(retrieved_chunks), component, relevant_component)
            output, _, _=self.run_llm_query(prompt)

            try:
                parsed_output = self.parse_json(output)
            except:
                parsed_output = self.try_again(prompt)

            with open(osp.join(save_dir,f"{component}.json"),"w", encoding="utf-8") as f:
                json.dump(parsed_output, f, indent=2, ensure_ascii=False)    

            if component=="macro_actions":
                relevant_component= parsed_output

        with open(osp.join(save_dir, "retrieval_log.json"), "w", encoding="utf-8") as f:
            json.dump(retrieval_log, f, indent=2, ensure_ascii=False)

        return save_dir

class Merge_RAG(Chunk_RAG):

    def build_chunk_embeddings(self, chapter_texts, book_idx, max_tokens=3000, overlap=100):
        chunk_embeddings = [] 
        for chapter_idx, (content_type,  chapter) in tqdm(enumerate(chapter_texts), total=len(chapter_texts), desc="Chunking"):
            chunks = self.chunk_text(chapter, max_tokens=max_tokens, overlap=overlap)
            for chunk_idx, chunk in enumerate(chunks):
                emb = self.get_embedding(chunk)
                chunk_embeddings.append(((book_idx, chapter_idx, chunk_idx), emb, chunk))
        return chunk_embeddings

    def retrieve_top_chunks(self, query, chunk_embeddings, top_k=3):
        query_emb = self.get_embedding(query)
        scores = []
        for (book_idx, chapter_idx, chunk_idx), emb, chunk in chunk_embeddings:
            sim = cosine_similarity(np.array(query_emb).reshape(1, -1), 
                                    np.array(emb).reshape(1, -1)).item()
            scores.append(((book_idx, chapter_idx, chunk_idx), sim, chunk))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

        retrieved = []
        for (book_idx, chapter_idx, chunk_idx), _, chunk in scores:
            retrieved.append(f"book{book_idx}_chapter{chapter_idx}_chunk{chunk_idx}\n{chunk}")
        return retrieved

    def extract(self, *, role: str, **kwargs) -> str:
        # Put outputs inside outputs/<domain>/...
        save_dir = osp.join(str(self.output_path), self.domain, self.__class__.__name__, "extracted_components", role)
        os.makedirs(save_dir, exist_ok=True)

        pdf_files = kwargs.get("pdf_files", None)
        if pdf_files is None:
            pdf_files = sorted(glob.glob(osp.join(str(self.textbooks_path), "*.pdf")))
        else:
            pdf_files = list(pdf_files)

        logger.info("[TOPA] Extracting components...")
        chunk_embeddings = []
        retrieval_log = {
            "method": "Merge_RAG",
            "role": role,
            "components": {}
        }
        for book_idx, book_path in enumerate(pdf_files):
            toc, chapter_texts = self.extract_chapters_and_toc(book_path)
            chunk_embeddings.extend(self.build_chunk_embeddings(chapter_texts, book_idx))
        
        relevant_component = []
        for component in tqdm(self.components[role], total=len(self.components[role])):

            retrieved_chunks = []
            for query in self.get_queries(component):
                retrieved_chunks.extend(self.retrieve_top_chunks(query, chunk_embeddings, top_k=30))
            retrieved_chunks = list(dict.fromkeys(retrieved_chunks))

            comp_log = {"retrieved": [], "chapter_counts": {}}
            for s in retrieved_chunks:
                pref = s.split('\n', 1)[0]
                mm = re.match(r"book(\d+)_chapter(\d+)_chunk(\d+)", pref)
                if mm:
                    b = int(mm.group(1))
                    ch = int(mm.group(2))
                    ck = int(mm.group(3))
                    key = f"{b}:{ch}"
                    comp_log["retrieved"].append({"book": b, "chapter": ch, "chunk": ck})
                    comp_log["chapter_counts"][key] = comp_log["chapter_counts"].get(key, 0) + 1
            retrieval_log["components"][component] = comp_log

            prompt = self.extract_component_prompt("\n".join(retrieved_chunks), component, relevant_component)
            output, _, _ = self.run_llm_query(prompt)

            try:
                parsed_output = self.parse_json(output)
            except:
                parsed_output = self.try_again(prompt)

            with open(osp.join(save_dir, f"{component}.json"),"w", encoding="utf-8") as f:
                json.dump(parsed_output, f, indent=2, ensure_ascii=False)    

            if component=="macro_actions":
                relevant_component= parsed_output

        with open(osp.join(save_dir, "retrieval_log.json"), "w", encoding="utf-8") as f:
            json.dump(retrieval_log, f, indent=2, ensure_ascii=False)

        return save_dir

class Rules(BaseExtractor):

    def extract_component_prompt(self, input_content, component, relevant_component=None):

        if component =="micro_actions":
            prompt=f"""
            You are an assistant that extracts {component} from provided rules extracted from {self.domain} textbooks.
            ### Input: 
            1. A list of **macro actions** extracted from a {self.domain} textbook.
            2. A summary of rules extracted from the textbook.

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
            prompt=f"""
            You are an assistant that extracts {component} from provided rules extracted from {self.domain} textbooks.
            
            ### Definition:
            {component} are {self.SPEC[component]["definition"]}

            ### Description:

            {self.SPEC[component]["description"]}
        
            ### Source Text:
            {input_content}

            ### Return the result as a **valid JSON object**, strictly following this format:  
            {self.SPEC[component]["format"]}

            ### Output:
            """
        return prompt
    
    def get_rules_prompt(self, chapter_text):
        prompt=f"""

        You are an expert assistant analyzing a {self.domain} textbook. Your goal is to extract **rules**  for dialogue modeling given a chapter. 
       
         **Rules**: 
        - the {self.domain} rules that the {self.system} should follow, strategies described in the form of: if conversation state / {self.user} state then action .

        ### Input Chapter Source Text:
        {chapter_text}

        - "if": a concise description of the situation, conversation state, or {self.user} state.
        - "then": a concise description of what the {self.system} should do next (action, constraint, or response strategy)

        ### Return a single string with the following structure:
        {self.SPEC["rules"]["format_str"]}

        ## Output:
        """
        return prompt


    def extract(self, *, role: str, **kwargs) -> str:
        pdf_path = Path(self.textbooks_path).expanduser()
        book_name = pdf_path.stem

        save_dir = osp.join(str(self.output_path), self.domain, self.__class__.__name__, "extracted_components", role, book_name)
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"[TOPA] Extracting components from book: {book_name}...")
        toc, chapter_texts = self.extract_chapters_and_toc(str(pdf_path))
        rules_str=""

        for i, (content_type, content) in tqdm(enumerate(chapter_texts), total=len(chapter_texts)):
            prompt=self.get_rules_prompt(content)
            output, _, _=self.run_llm_query(prompt)
            rules_str+= output
        
        relevant_component=[]
        for component in self.components[role]:

            prompt = self.extract_component_prompt(rules_str, component, relevant_component)
            output, _, _=self.run_llm_query(prompt)

            try:
                parsed_output = self.parse_json(output)
            except:
                parsed_output = self.try_again(prompt)

            with open(osp.join(save_dir,f"{component}.json"),"w", encoding="utf-8") as f:
                json.dump(parsed_output, f, indent=2, ensure_ascii=False)    

            if component=="macro_actions":
                relevant_component= parsed_output

        return save_dir

class Mamba(Chap_Seq):
    def extract(self, *args, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba-Mini-1.7",
                                                          dtype=torch.bfloat16,
                                                          attn_implementation="sdpa",
                                                          #  attn_implementation="flash_attention_2",
                                                          device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba-Mini-1.7")

        return super().extract(*args, **kwargs)

    def run_llm_query(self, prompt):
        t0 = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device) # torch.Size([1, 211140])
        outputs = self.model.generate(**inputs, max_new_tokens=8192, temperature=0.4, top_p=0.95)

        output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        tok_in = inputs["input_ids"].shape[1]
        tok_out = len(output_ids)
        lat = time.time() - t0
        self._record_usage(tok_in, tok_out, lat, tag="local")
        return response, tok_in, tok_out

class ManualPromptExporter(BaseExtractor):
    """
    Manual extraction: write prompts (.txt) + create output placeholders (.json).
    No textbooks. Prompts ask the model to generate components from general domain knowledge.
    """

    MODEL_TAG = "manual"

    def _component_prompt(self, component: str, role: str) -> str:
        prompt = f"""You are an assistant helping design a hierarchical MDP for {self.domain} conversations."""

        if component == "micro_actions":
            prompt += f"""\nYour goal is to extract **Action Space (Macro + Micro Actions)** relevant to intervention modeling: 
macro actions are:
{self.SPEC["macro_actions"]["definition"]}
{self.SPEC["macro_actions"]["description"]}

micro actions are:
{self.SPEC["micro_actions"]["definition"]}
{self.SPEC["micro_actions"]["description"]}"""
        else:
            if role == "system":
                prompt += f"\nYour goal is to extract {component} relevant to intervention modeling.\n"
            else:
                prompt += f"\nYour goal is to extract {component} relevant to **{self.user} modeling** in {self.domain} interventions.\n"

            prompt += f"""
### Definition:
{component} are {self.SPEC[component]["definition"]}

{"### Description:" if self.SPEC[component]["description"] != "" else ""}
{self.SPEC[component]["description"]}"""
    
        prompt += f"""
### Return the result as a **valid JSON object**, strictly following this format:  
{self.SPEC[component]["format"]}

### Output:"""
        return prompt

    def _load_outputs_if_ready(self, outputs_dir: Path, role: str) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for component in self.components[role]:
            p = outputs_dir / f"{component}.json"
            if not p.exists() or p.read_text(encoding="utf-8").strip() == "":
                raise RuntimeError(f"Missing/empty output for component '{component}'. Expected file:\n  {p}")
            merged[component] = json.loads(p.read_text(encoding="utf-8"))
        return merged

    def _missing_outputs(self, outputs_dir: Path, role: str) -> List[Path]:
        missing = []
        for component in self.components[role]:
            if component == "macro_actions":
                continue
            p = outputs_dir / f"{component}.json"
            if (not p.exists()) or (p.read_text(encoding="utf-8").strip() == ""):
                missing.append(p)
        return missing

    def extract(self, *, role: str, **kwargs: Any) -> Dict[str, Any]:

        base_dir = Path(self.output_path) / self.domain / self.MODEL_TAG / role
        prompts_dir = base_dir / "prompts"
        outputs_dir = base_dir / "outputs_expected"

        prompts_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Write one prompt per component (+ placeholder output files)
        for component in self.components[role]:
            if component == "macro_actions":
                continue
            (prompts_dir / f"{component}.txt").write_text(self._component_prompt(component, role), encoding="utf-8")
            out_path = outputs_dir / f"{component}.json"
            if not out_path.exists():
                out_path.write_text("", encoding="utf-8")

        logger.info(f"[TOPA][{self.MODEL_TAG}] Prompts written to:\n  {prompts_dir}")
        logger.info(f"[TOPA][{self.MODEL_TAG}] Paste model outputs into:\n  {outputs_dir}")

        # If user already pasted outputs, load + return merged components
        try:
            self._load_outputs_if_ready(outputs_dir, role)
            return str(outputs_dir)
        except:
            pass

        # Otherwise, block until user confirms and outputs are ready
        while True:
            cmd = input(
                "\nType 'done' after you paste outputs, 'list' to show missing files, or 'exit' to stop: "
            ).strip().lower()

            if cmd in ("list", "ls"):
                missing = self._missing_outputs(outputs_dir, role)
                if not missing:
                    print("[TOPA] All output files exist and are non-empty.")
                else:
                    print("[TOPA] Still missing/empty:")
                    for p in missing:
                        print(f"  - {p}")
                continue

            if cmd in ("done", "d"):
                missing = self._missing_outputs(outputs_dir, role)
                if missing:
                    print("[TOPA] Not ready yet. Missing/empty files:")
                    for p in missing:
                        print(f"  - {p}")
                    continue
                return str(outputs_dir)

            if cmd in ("exit", "quit", "q"):
                return str(outputs_dir)

            print("[TOPA] Unknown command. Use: done | list | exit")

class ChatGPT(ManualPromptExporter):
    MODEL_TAG = "ChatGPT"

class DeepSeek(ManualPromptExporter):
    MODEL_TAG = "DeepSeek"