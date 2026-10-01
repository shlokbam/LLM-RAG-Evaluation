import os
import re
import glob
import json
import time
import random
import asyncio
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

load_dotenv()

MODEL_NAME = "open-mistral-7b"


import json
from json_repair import repair_json


def clean_json_output(res: str) -> str:
    """Strips preambles, markdown codeblocks, extra trailing blocks, and returns a single valid JSON string."""
    if not isinstance(res, str):
        return res
    res = res.strip()

    # Extract first complete `{ ... }` JSON object using brace stack tracking
    first_b = res.find("{")
    if first_b != -1:
        stack = 0
        end_b = -1
        for i in range(first_b, len(res)):
            if res[i] == "{":
                stack += 1
            elif res[i] == "}":
                stack -= 1
                if stack == 0:
                    end_b = i
                    break
        if end_b != -1:
            res = res[first_b:end_b + 1]

    try:
        parsed = repair_json(res, return_objects=True)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed)
    except Exception:
        pass

    return res


# --- Custom Mistral Model Wrapper for DeepEval ---
class MistralLLM(DeepEvalBaseLLM):
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = ChatMistralAI(
            model=model_name,
            api_key=os.getenv("MISTRAL_API_KEY"),
            max_retries=5,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        if schema:
            prompt = f"{prompt}\n\nIMPORTANT: Respond strictly with a valid JSON object matching the requested schema without any markdown formatting or extra text."
        for attempt in range(5):
            try:
                res = self.model.invoke(prompt).content
                return clean_json_output(res)
            except Exception as e:
                if ("429" in str(e) or "rate" in str(e).lower()) and attempt < 4:
                    time.sleep(3 * (attempt + 1))
                else:
                    raise e

    async def a_generate(self, prompt: str, schema=None, *args, **kwargs) -> str:
        if schema:
            prompt = f"{prompt}\n\nIMPORTANT: Respond strictly with a valid JSON object matching the requested schema without any markdown formatting or extra text."
        for attempt in range(5):
            try:
                res = await self.model.ainvoke(prompt)
                return clean_json_output(res.content)
            except Exception as e:
                if ("429" in str(e) or "rate" in str(e).lower()) and attempt < 4:
                    await asyncio.sleep(3 * (attempt + 1))
                else:
                    raise e

    def get_model_name(self):
        return self.model_name


# --- reuse VTT cleaning + chunking (same as the retriever) ---
def load_chunks():
    texts = []
    for path in glob.glob("Data/*.vtt"):
        with open(path, encoding="utf-8") as f:
            lines = [
                ln.strip()
                for ln in f
                if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln
            ]
        texts.append(" ".join(lines))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))


# --- generate ---
chunks = load_chunks()
sample = random.sample(chunks, min(6, len(chunks)))
contexts = [[c] for c in sample]

mistral_llm = MistralLLM(model_name=MODEL_NAME)
synthesizer = Synthesizer(model=mistral_llm)
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=1,
)


# --- convert to YOUR schema (id / query / ideal_answer / source) ---
rows = []
for i, g in enumerate(goldens, 1):
    rows.append(
        {
            "id": f"g{i:03d}",
            "query": g.input,
            "ideal_answer": g.expected_output,
            "source": "TODO-verify",
        }
    )

output_path = "Goldens/retriever_deepeval_goldens.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"wrote {len(rows)} DRAFT goldens -> {output_path}")
print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")