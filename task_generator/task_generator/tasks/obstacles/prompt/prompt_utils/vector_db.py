import json
import time

import chromadb
import numpy as np
from arena_simulation_setup.utils.llm import genai
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_CFG = genai.types.EmbedContentConfig(task_type="retrieval_document", output_dimensionality=768)
COLLECTION_NAME = "bt-nodes-reference"


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, client: genai.Client, model: str, config: genai.types.EmbedContentConfig, *args: object, **kwargs: object) -> None:
        self.client = client
        self.model = model
        self.config = config

    def __call__(self, input_: Documents) -> Embeddings:
        docs: list[str] = input_
        response = self.client.models.embed_content(model=self.model, contents=list(docs), config=self.config)
        assert response.embeddings is not None
        return [np.array(item.values) for item in response.embeddings if item.values is not None]


def process_json_doc(doc_path: str) -> list[str]:
    with open(doc_path) as f:
        data = json.load(f)

    documents = []

    for category, items in data["Node categories"].items():
        for node in items:
            doc = ""
            doc = f"\nNode name: {node['name']}\nPurpose: {node['purpose']}"
            doc += "\nInputs:"
            for input_ in node["inputs"]:
                input_name = input_["name"]
                input_type = input_["type"]
                input_description = input_["description"]
                doc += f"\n\t- {input_name} ({input_type}): {input_description}"
            doc += "\nOutputs:"
            for output in node["outputs"]:
                output_name = output["name"]
                output_type = output["type"]
                output_description = output["description"]
                doc += f"\n\t- {output_name} ({output_type}): {output_description}"
            node_type = node["n_agent"]
            doc += f"\nMetadata:\n\t-Category: {category}\n\t-Node type: {node_type}"
            documents.append(doc)

    return documents


def create_chroma_db(documents: list[str], db_path: str, client: genai.Client, collection_name: str = COLLECTION_NAME, embedding_model: str = EMBEDDING_MODEL, embedding_config: genai.types.EmbedContentConfig = EMBEDDING_CFG) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(db_path)

    collection = chroma_client.create_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction(client=client, model=embedding_model, config=embedding_config))

    initiali_size = collection.count()
    for i, d in tqdm(enumerate(documents), total=len(documents), desc=f"Creating Chroma Collection at {db_path}"):
        collection.add(documents=d, ids=str(i + initiali_size))
        time.sleep(0.5)

    return collection


def get_chroma_collection(db_path: str, client: genai.Client, collection_name: str = COLLECTION_NAME, embedding_model: str = EMBEDDING_MODEL, embedding_config: genai.types.EmbedContentConfig = EMBEDDING_CFG) -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(db_path)

    return chroma_client.get_collection(name=collection_name, embedding_function=GeminiEmbeddingFunction(client=client, model=embedding_model, config=embedding_config))


def get_relevant_bt_nodes(query: str, collection: chromadb.Collection, n_results: int = 10) -> str:
    result = collection.query(query_texts=[query], n_results=n_results)
    assert result["documents"] is not None
    passages = result['documents'][0]

    nodes_descriptions = "\n".join(("", *passages))

    return nodes_descriptions.encode("utf-8").decode("unicode_escape").strip()


if __name__ == "__main__":
    from arena_hunav_sim_bridge import CHROMA_DB_PATH  # pyright: ignore[reportMissingImports]

    inference_client = genai.Client()
    chroma_collection = get_chroma_collection(CHROMA_DB_PATH, inference_client)

    prompt = "A group of 5 constructors rapidly organize themselves into a queue in the main central hallway, by the reception room door. As soon as a spot opens at the front, each person immediately steps forward, advancing in sequence toward the waiting area door. After reaching the front of the line, one person waits 20 seconds, then he enters the reception room, toward the reception counter, then he goes to main waiting area and find a seat that hasn't been taken. The lines continuously compress and move forward as people shuffle ahead whenever the person in front moves."

    bt_nodes = get_relevant_bt_nodes(
        query=f"What are the nodes should be used for creating the behavior tree as described below: \"{prompt}\". Use GoTo node to guide agents to isolated places if needed.",
        collection=chroma_collection,
    )

    print(bt_nodes)
