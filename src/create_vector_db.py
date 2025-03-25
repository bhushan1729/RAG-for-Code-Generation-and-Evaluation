import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents.base import Document

def get_embedding_model():
    '''Returns the embedding model using HuggingFaceBgeEmbeddings with GPU support.'''
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True}
    model_norm = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},  # Change 'cpu' to 'cuda' for GPU
        encode_kwargs=encode_kwargs
    )
    return model_norm

def create_vectordb(file_path: str, vector_db_path: str):
    '''Creates and saves the vector database.'''
    print("Creating vector database...")

    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()

    # Split the text by two newlines to separate each test case
    split_text = text.split('\n\n')

    obj_list = [Document(page_content=chunk.strip()) for chunk in split_text if chunk.strip()]

    # Create the vector database using the embedding model
    model_norm = get_embedding_model()
    db = FAISS.from_documents(documents=obj_list, embedding=model_norm)
    db.save_local(vector_db_path)
    print("Vector database created and saved successfully.")

if __name__ == "__main__":
    data_file = "data/test_steps.txt"
    vector_db_dir = "vector_database"
    os.makedirs(vector_db_dir, exist_ok=True)
    vector_db_path = os.path.join(vector_db_dir, "faiss_index")

    create_vectordb(data_file, vector_db_path)