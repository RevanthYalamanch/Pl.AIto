import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma 

FILE_PATH = "scripture_text.txt"
INDEX_PATH = "bhagavad_gita_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "bhagavad_gita_collection"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def create_knowledge_base():
    if os.path.exists(INDEX_PATH):
        return

    try:
        loader = TextLoader(FILE_PATH, encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        print(f"Error: File '{FILE_PATH}' not found. Please create it and paste your scripture text inside.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} 
    )
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=INDEX_PATH, 
        collection_name=CHROMA_COLLECTION_NAME
    )
    vector_store.persist()
    print(f"Successfully created and saved knowledge base to '{INDEX_PATH}'.")


if __name__ == "__main__":
    create_knowledge_base()