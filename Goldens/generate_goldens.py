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

    def generate(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                return self.model.invoke(prompt).content
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise e

    async def a_generate(self, prompt: str) -> str:
        for attempt in range(5):
            try:
                res = await self.model.ainvoke(prompt)
                return res.content
            except Exception as e:
                if "429" in str(e) and attempt < 4:
                    await asyncio.sleep(2 * (attempt + 1))
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