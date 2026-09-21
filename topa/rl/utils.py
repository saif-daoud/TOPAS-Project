import time
import json
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from openai import OpenAI, AzureOpenAI
import torch
from transformers import AutoModel

def build_logger():
    # disable noisy libs globally
    for name in [
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.connection_pool",
        "openai",
        "pydot"
    ]:
        l = logging.getLogger(name)
        l.propagate = False
        l.handlers.clear()
        l.addHandler(logging.NullHandler())

    return logging.getLogger("topa")

logger = build_logger()

def parse_json(output: str):
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

def run_llm_query(prompt, api_key):
    api_version = "2024-12-01-preview"
    endpoint = "https://qcri-sakina.cognitiveservices.azure.com/"
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )
    DEPLOYMENT_NAME = "gpt-4.1"

    flag = True
    while flag:
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            flag = False
        except Exception as e:
            logger.info(f"Some error happened here. {e}")
            time.sleep(5)

    output_text = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return output_text, input_tokens, output_tokens

def get_llm_response(prompt, api_key):
    api_version = "2024-12-01-preview"
    endpoint = "https://qcri-sakina.cognitiveservices.azure.com/"
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    DEPLOYMENT_NAME = "gpt-4.1"
    
    flag = True
    while flag:
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            flag = False
        except Exception as e:
            logger.info(f"Some error happened here. {e}")
            time.sleep(5)
    return response.choices[0].message.content

def get_llm_response_messages(prompt, api_key, max_tokens: int = 128, temperature: float = 0, n: int = 1):
    api_version = "2024-12-01-preview"
    endpoint = "https://qcri-sakina.cognitiveservices.azure.com/"
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    DEPLOYMENT_NAME = "gpt-4.1"

    if isinstance(prompt, list):
        chat_prompt = prompt
    elif isinstance(prompt, str):
        chat_prompt = [
            {"role": "user", "content": prompt}
        ]
    else:
        raise Exception(f"Invalid prompt format: {prompt}")
    
    flag = True
    while flag:
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=chat_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
            )
            if n == 1:
                output = response.choices[0].message.content
                if output is None:
                    logger.info("output None")
                    continue
                output = output.strip()
            else:
                output = []
                for choice in response.choices:
                    out = choice.message.content
                    if out is None:
                        continue
                    output.append(out.strip())
                if len(output) == 0:
                    logger.info("All outputs are None !")
                    continue
            flag = False
        except Exception as e:
            logger.info(f"Some error happened here. {e}")
            time.sleep(5)
    return output

# ========================
# Embedder helpers
# ========================
class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def embed_many(self, texts: List[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype=np.float32)

class JinaLocalEmbedder:
    """
    Local wrapper for jinaai/jina-embeddings-v4.
    - For RAG, use prompt_name="query" for queries, prompt_name="passage" for docs.
    """

    def __init__(
        self,
        model_id: str = "jinaai/jina-embeddings-v4",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        task: str = "retrieval",
        prompt_name: Optional[str] = "query",
        truncate_dim: Optional[int] = None,
        batch_size: int = 8,
        return_multivector: bool = False
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self.task = task
        self.prompt_name = prompt_name
        self.truncate_dim = truncate_dim
        self.batch_size = batch_size
        self.return_multivector = return_multivector

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)

        # Disable jina-embeddings-v4 tqdm progress bar
        if hasattr(self.model, "verbosity"):
            self.model.verbosity = 0
        if hasattr(self.model, "config") and hasattr(self.model.config, "verbosity"):
            self.model.config.verbosity = 0
        for obj in (self.model, getattr(self.model, "model", None)):
            if obj is not None and hasattr(obj, "verbosity"):
                obj.verbosity = 0

        self.model.eval()

    def _encode_texts(self, texts: List[str]):
        return self.model.encode_text(
            texts=texts,
            task=self.task,
            prompt_name=self.prompt_name,
            truncate_dim=self.truncate_dim,
            batch_size=len(texts), # we batch outside
            return_multivector=self.return_multivector,
        )

    @staticmethod
    def _to_numpy(out) -> np.ndarray:
        stacked = torch.stack([t.detach().to("cpu") for t in out], dim=0)
        return stacked.float().numpy()

    def embed(self, text: str) -> np.ndarray:
        X = self.embed_many([text], show_progress=False)
        return X[0].astype(np.float32, copy=False)

    def embed_many(self, texts, show_progress: bool = True):
        n = len(texts)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        bs = max(1, self.batch_size)
        batches = range(0, n, bs)

        pbar = tqdm(total=n, desc="Embedding", unit="text", disable=not show_progress)

        all_vecs = []
        with torch.inference_mode():
            for i in batches:
                chunk = texts[i : i + bs]
                out = self._encode_texts(chunk)
                X = self._to_numpy(out).astype(np.float32, copy=False)

                if X.ndim == 1:
                    X = X[None, :]

                all_vecs.append(X)
                pbar.update(len(chunk))
        pbar.close()

        return np.concatenate(all_vecs, axis=0)
    
def process_dialogue(dialogue: List[dict], domain_params: dict):
    system = domain_params["system"]
    user = domain_params["user"]

    utterances = []
    sys_c = 0
    usr_c = 0
    for turn in dialogue:
        if turn["speaker"] == "system":
            utt_id = f"{system}_{sys_c}"
            sys_c += 1
            speaker = system
        elif turn["speaker"] == "user":
            utt_id = f"{user}_{usr_c}"
            usr_c += 1
            speaker = user
        else:
            raise Exception(f"Unkown speaker {turn['speaker']} - Utterance : {turn}")
        utterances.append({"utterance_id": utt_id, "speaker": speaker, "utterance": turn["text"]})
    return utterances

def build_session_index(data: list, domain_params: dict):
    """Build (user_idx, session_idx) -> list[utterances], plus id->pos map."""
    sessions = {}
    pos = {}
    for user_idx, user in enumerate(data):
        for session_idx, session in enumerate(user["sessions"]):
            utterances = process_dialogue(session["dialogue"], domain_params)
            sessions[(user_idx, session_idx)] = utterances
            pos[(user_idx, session_idx)] = {u["utterance_id"]: i for i, u in enumerate(utterances)}
    return sessions, pos