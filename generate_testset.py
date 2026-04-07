import os
import sys
import random
import asyncio
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = "ollama"

import json
import pandas as pd
from ragas.testset import TestsetGenerator
from ragas.testset.transforms import Extractor
from ragas.testset.transforms.extractors import (
    HeadlinesExtractor, 
    SummaryExtractor, 
    NERExtractor
)
from ragas.testset.transforms import SummaryCosineSimilarityBuilder
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.run_config import RunConfig
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from audit_engine import get_clean_chunks 

# Force Windows to use SelectorEventLoop to prevent "Event loop is closed"
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Custom Embedding Logic to bypass the 'Skip' bug
class RagasOllamaEmbeddings(BaseRagasEmbeddings):
    def __init__(self, langchain_embeddings):
        super().__init__()
        self.langchain_embeddings = langchain_embeddings
    def embed_query(self, text): return self.langchain_embeddings.embed_query(text)
    def embed_documents(self, texts): return self.langchain_embeddings.embed_documents(texts)
    async def aembed_query(self, text): return await self.langchain_embeddings.aembed_query(text)
    async def aembed_documents(self, texts): return await self.langchain_embeddings.aembed_documents(texts)
    async def embed_text(self, text): return await self.langchain_embeddings.aembed_query(text)

# Custom Extractor to bypass the 'Skip existing property' bug
class ForceEmbeddingExtractor(Extractor):
    def __init__(self, property_name, embedding_model):
        super().__init__()
        self.property_name = property_name
        self.embedding_model = embedding_model
    
    def filter_nodes(self, nodes):
        # Must be sync (def) to avoid the RuntimeWarning
        return nodes

    async def extract(self, node):
        # In Ragas v0.4+, the data is stored in the .properties dict
        # We check both the specific property and 'page_content' (the default text)
        if self.property_name == "page_content":
            # Ragas usually stores the main text under 'page_content' or 'content' in properties
            text = node.properties.get("page_content") or node.properties.get("content")
        else:
            text = node.properties.get(self.property_name)

        if not text:
            return None
            
        embedding = await self.embedding_model.embed_text(text)
        return f"{self.property_name}_embedding", embedding

# 1. Setup Models using the modern Factory Pattern
# Uses the OpenAI-compatible endpoint of Ollama
print("Initializing Local Models...")
generator_llm = llm_factory(
    model="llama3.1:8b", 
    client=OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
)

# Using OpenAIEmbeddings pointed at Ollama to avoid factory deprecations
raw_embeddings = OllamaEmbeddings(model="llama3.2:1b")
generator_embeddings = RagasOllamaEmbeddings(raw_embeddings)

# 2. Load & Clean Documents using your own ETL
loader = DirectoryLoader("data/regulations/", glob="./*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()
all_chunks = get_clean_chunks(docs, chunk_size=1500, chunk_overlap=300)
clean_chunks = random.sample(all_chunks, min(30, len(all_chunks)))

# 3. Complete transform pipeline
transforms = [
    HeadlinesExtractor(llm=generator_llm),
    SummaryExtractor(llm=generator_llm),
    NERExtractor(llm=generator_llm),  # This populates the 'entities' property
    ForceEmbeddingExtractor(property_name="page_content", embedding_model=generator_embeddings),
    ForceEmbeddingExtractor(property_name="summary", embedding_model=generator_embeddings),
    SummaryCosineSimilarityBuilder(threshold=0.5)
]

# 4. Initialize Generator
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
config = RunConfig(max_workers=1, timeout=300)

async def main():
    print(f"Starting Fail-Safe Generation (Sample Size: {len(clean_chunks)} chunks).")
    try:
        # Explicitly use SingleHop synthesizer for maximum local stability
        simple_synth = SingleHopSpecificQuerySynthesizer(llm=generator_llm)

        testset = generator.generate_with_chunks(
            chunks=clean_chunks, 
            testset_size=5, 
            transforms=transforms,
            query_distribution=[(simple_synth, 1.0)],
            run_config=config
        )
        testset.to_pandas().to_json("synthetic_golden_dataset.json", orient="records", indent=4)
        print("Success! Testset saved.")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    asyncio.run(main())