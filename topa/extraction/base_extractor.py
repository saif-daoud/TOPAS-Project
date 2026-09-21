from abc import ABC, abstractmethod 
from openai import AzureOpenAI
from collections import Counter
import fitz
import json
import os
import time

from .specs import (
    macro_action_definition,
    macro_action_description,
    macro_action_format,
    macro_action_format_str,
    micro_action_definition,
    micro_action_description,
    micro_action_format,
    micro_action_format_str,
    conversation_states_definition,
    conversation_states_description,
    conversation_states_format,
    conversation_states_format_str,
    knowledge_graph_definition,
    knowledge_graph_description,
    knowledge_graph_format,
    knowledge_graph_format_str,
    cautions_definition,
    cautions_description,
    cautions_format,
    cautions_format_str,
    rules_definition,
    rules_format,
    rules_format_str,
    user_action_definition,
    user_action_format,
    user_profile_definition,
    user_profile_description,
    user_profile_format,
    user_profile_format_str,
)

from ..utils import build_logger
logger = build_logger()

class TOCNode:
    def __init__(self, level, title):
        self.level = level
        self.title = title
        self.children = []

    def to_dict(self):
        return {
            "title": self.title,
            "level": self.level,
            "children": [child.to_dict() for child in self.children],
        }

class BaseExtractor(ABC):

    def __init__(self, domain_name, system, user, interaction_unit, textbooks_path, output_path, api_key, mode):
        self.output_path = output_path
        self.domain = domain_name
        self.system = system
        self.user = user
        self.interaction_unit = interaction_unit
        self.textbooks_path = textbooks_path
        self.api_key = api_key
        self.mode = mode

        # Usage tracking (for MEM/TOK/LAT/COST metrics)
        self._usage = {"tok_in": 0, "tok_out": 0, "lat_sec": 0.0, "mem_peak": 0}
        self._usage_calls = []
        self.components = {
            "system":["macro_actions", "micro_actions", "conversation_states", "knowledge_graph","cautions", "rules"],
            "user":["user_profile", "user_actions"]
        }

        if self.mode == "dialogue":
            _macro_def = macro_action_definition
            _macro_desc = macro_action_description
            _micro_def = micro_action_definition
            _micro_desc = micro_action_description
            _state_def = conversation_states_definition
            _state_desc = conversation_states_description
            _kg_def = knowledge_graph_definition
            _kg_desc = knowledge_graph_description
            _cautions_def = cautions_definition
            _cautions_desc = cautions_description
        else:
            raise ValueError(f"Not implemented mode: {self.mode}")

        self.SPEC = {
            "macro_actions": {
                "definition": _macro_def.format(
                    system=self.system, user=self.user
                ),
                "description":_macro_desc.format(
                    system=self.system, user=self.user
                ),
                "format": macro_action_format,
                "format_str": macro_action_format_str
            },

            "micro_actions": {
                "definition": _micro_def.format(
                    system=self.system, user=self.user
                ),
                "description":_micro_desc.format(
                    system=self.system, user=self.user
                ),
                "format": micro_action_format,
                "format_str": micro_action_format_str,
            },

            "conversation_states": {
                "definition": _state_def.format(
                    system=self.system,
                    domain=self.domain,
                    interaction_unit=self.interaction_unit
                ),
                "description":_state_desc.format(
                     interaction_unit=self.interaction_unit
                ),
                "format": conversation_states_format,
                "format_str": conversation_states_format_str
            },

            "knowledge_graph": {
                "definition": _kg_def.format(
                    interaction_unit=self.interaction_unit,
                    user=self.user
                ),
                "description":_kg_desc,
                "format": knowledge_graph_format,
                "format_str": knowledge_graph_format_str
            },

            "cautions": {
                "definition": _cautions_def.format(
                    system=self.system,
                    user=self.user,
                    domain=self.domain,
                    interaction_unit=self.interaction_unit
                ),
                "description": _cautions_desc,
                "format": cautions_format,
                "format_str": cautions_format_str
            },

            "rules": {
                "definition": rules_definition.format(
                    system=self.system,
                    user=self.user
                ),
                "description": "",
                "format": rules_format,
                "format_str": rules_format_str
            },

            "user_actions": {
                "definition": user_action_definition.format(
                    user=self.user,
                    system=self.system
                ),
                "description": "",
                "format": user_action_format
            },

            "user_profile": {
                "definition": user_profile_definition.format(
                    user=self.user,
                    domain=self.domain,
                    interaction_unit=self.interaction_unit,
                    system=self.system
                ),
                "description":user_profile_description,
                "format": user_profile_format,
                "format_str": user_profile_format_str
            }
        }

    @abstractmethod
    def extract(self):

        """Return extracted knowledge."""
        pass

    @staticmethod
    def detect_chapter_level(toc):

        level_counter = Counter()
        for level, title, _ in toc:
            if title.lower().startswith("chapter"):
                level_counter[level] += 1
        
        if not level_counter:
            return 1, False  # No "Chapter" entries found, use level 1
        
        # Return the level with the most "chapter" entries
        most_common_level, _ = level_counter.most_common(1)[0]
        return most_common_level, True
    
    @staticmethod
    def build_tree(toc, book_name, limit_level=None):
        root = TOCNode(0, book_name)
        stack = [root]

        for level, title, _ in toc:
            if limit_level is not None and level > limit_level:
                continue
            node = TOCNode(level, title)
            
            # Find the correct parent
            while stack and stack[-1].level >= level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)

        return root.to_dict()
    
    @staticmethod
    def get_titles_str(node, indent=0):
        result = "\t" * indent + f"{node['title']}\n"
        for child in node['children']:
            result += BaseExtractor.get_titles_str(child, indent + 1)
        return result

    @classmethod
    def extract_chapters_and_toc(cls, book_path):
        doc = fitz.open(book_path)
        toc = doc.get_toc()  # [level, title, page]

        # Filter ToC entries
        chapter_level, use_chapter = cls.detect_chapter_level(toc)
        # chapters = [(title, page - 1) for level, title, page in toc if level == chapter_level]

        chapters = []
        for level, title, page in toc:
            if level == chapter_level:
                if not use_chapter:
                    chapters.append((title, page - 1))
                elif title.lower().startswith("chapter"):
                    chapters.append((title, page - 1))
        
        # Create a list to hold chapter text tuples: (title, full_text)
        chapter_texts = []

        for i, (title, start_page) in enumerate(chapters):
            # Determine end page: it's the start of the next chapter, or end of document
            if i + 1 < len(chapters):
                end_page = chapters[i + 1][1]
            else:
                end_page = len(doc)

            # Extract text for the chapter
            text = ""
            for page_num in range(start_page, end_page):
                text += doc[page_num].get_text()

            chapter_texts.append((title, text.strip()))

        # Get TOC
        structured_toc = cls.build_tree(toc, book_path.rstrip(".pdf"))
        structured_toc_str = cls.get_titles_str(structured_toc)

        return structured_toc_str, chapter_texts

    def string_to_json_prompt(self, role, extracted_text):

        prompt = f"""
        You are an expert assistant.  
        You are given the following **extracted string components** from a {self.domain} textbook:  
        ---
        {extracted_text}
        ---

        Your task: Convert this text into a **single valid JSON object** with exactly these top-level keys:
        """
        prompt+="{ \n"

        for component in self.components[role]:
            prompt+=f"""
            "{component}":
             {self.SPEC[component]["format"]}
            """
        prompt+="} \n"
        prompt+="- Ensure the JSON is valid and properly formatted."
        return prompt
    
    def reset_usage(self):
        self._usage = {"tok_in": 0, "tok_out": 0, "lat_sec": 0.0, "mem_peak": 0}
        self._usage_calls = []

    def get_usage(self):
        return dict(self._usage)

    def save_usage(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "usage.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"usage": self._usage, "calls": self._usage_calls}, f, indent=2, ensure_ascii=False)
        return p

    def _record_usage(self, tok_in: int, tok_out: int, lat_sec: float, tag: str = "llm"):
        self._usage["tok_in"] += tok_in
        self._usage["tok_out"] += tok_out
        self._usage["lat_sec"] += lat_sec
        peak = tok_in + tok_out
        if peak > self._usage["mem_peak"]:
            self._usage["mem_peak"] = peak
        self._usage_calls.append({
            "tag": tag,
            "tok_in": tok_in,
            "tok_out": tok_out,
            "lat_sec": lat_sec,
        })

    def run_llm_query(self, prompt):
    
        api_key = self.api_key
        api_version = "2024-12-01-preview"
        endpoint = "https://qcri-sakina.cognitiveservices.azure.com/"
        
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )

        # est_input_tokens = num_tokens_from_messages(prompt, model="gpt-4.1")
        # logger.info(f"est_input_tokens: {est_input_tokens}")

        DEPLOYMENT_NAME = "gpt-4.1"
        error_n = 0
        t0 = time.time()
        while True:
            if error_n == 2:
                return "None", 0, 0
            try:
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                logger.info(f"Some error happened here. {e}")
                error_n += 1
                time.sleep(60)

        lat = time.time() - t0
        output_text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        self._record_usage(input_tokens, output_tokens, lat, tag="llm")
        return output_text, input_tokens, output_tokens
    
    def try_again(self, prompt_for_annotation):
        logger.info(f"[WARNING] Unable to parse output, try again")
        output,_,_= self.run_llm_query(prompt_for_annotation)
        output = self.parse_json(output)
        return output
    
    @staticmethod
    def parse_json(output):
        cleaned_output = output.strip()
        start_tag = "```json"
        end_tag = "```"

        start_index = cleaned_output.find(start_tag)
        if start_index == -1:
            start_index = 0
        else:
            start_index += len(start_tag)

        cleaned_output = cleaned_output[start_index:]

        end_index = cleaned_output.rfind(end_tag)
        if end_index != -1:
            cleaned_output = cleaned_output[:end_index]
        
        return json.loads(cleaned_output.strip())


# import tiktoken

# def num_tokens_from_messages(message, model="gpt-4.1"):
#     """Estimate tokens used by a list of chat messages."""
#     try:
#         encoding = tiktoken.encoding_for_model(model)
#     except KeyError:
#         encoding = tiktoken.get_encoding("o200k_base")  # works for gpt-4.1 family

#     return len(encoding.encode(message))