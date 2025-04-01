import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./amazon_product"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())
async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


custom_prompt = """
You are an expert assistant in web development and state management, specializing in mapping Redux-like state objects to corresponding DOM elements. 
Provide detailed and structured answers with practical examples that explain how specific state changes should trigger UI updates and endpoint modifications. 

---Conversation History--- 
{history} 

---Knowledge Base--- 
{context_data} 

---Response Rules---

Target format and length: {response_type}
"""


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        with open("dataset/amazon_product.html", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # # Perform naive search
        # print(
        #     await rag.aquery(
        #         "Extract from the provided DOM all HTML elements that are likely candidates for the selectedProduct state key. Focus on elements that serve as search inputs such as input fields, search forms, or elements with attributes id, name, class, placeholder containing keywords like search or query. Return a JSON object mapping selectedProduct to an array of objects each detailing the element's tag, relevant attributes, and inner text if applicable", 
        #         param=QueryParam(mode="naive"),
        #         system_prompt=custom_prompt  # Pass the custom prompt
        #     )
        # )

        # print("\n -------- \n")


        # Perform local search
        print(
            await rag.aquery(
                "Retrieve all DOM elements and UI components associated with the state object key: selectedProduct.  Identify which UI elements (e.g., icons, containers and endpoints) should be updated when this state key changes. Return a JSON object mapping the state object key: selectedProduct to an array of objects each detailing the line within the DOM, element's tag, relevant attributes, and inner text if applicable.", 
                param=QueryParam(mode="local"),
                 system_prompt=custom_prompt  # Pass the custom prompt
            )
        )
        print("\n -------- \n")

        # Perform global search
        print(
            await rag.aquery(
                "Retrieve all DOM elements and UI components associated with the state object key: selectedProduct. Identify which UI elements (e.g., icons, containers) and endpoints should be updated when this state key changes, detailing the specific modifications required.",
                param=QueryParam(mode="global"),
                system_prompt=custom_prompt  # Pass the custom prompt
            )
        )
        print("\n -------- \n")

        # Perform hybrid search
        print(
            await rag.aquery(
                "Retrieve all DOM elements and UI components associated with the state object key: selectedProduct.  Identify which UI elements (e.g., icons, containers and endpoints) should be updated when this state key changes. Return a JSON object mapping the state object key: selectedProduct to an array of objects each detailing the line within the DOM, element's tag, relevant attributes, and inner text if applicable.",
                param=QueryParam(mode="hybrid"),
                system_prompt=custom_prompt  # Pass the custom prompt
            )
        )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
