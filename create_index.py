import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings

FILE_PATH = "easwaran.txt"
CONNECTION_STRING = "postgresql+psycopg2://postgres:acakdoir@192.168.2.221:5433"
COLLECTION_NAME = "bhagavad_gita_collection"
GPU_SERVER_URL = "http://192.168.2.221:11434"

def clean_text_for_embedding(documents):
    cleaned_documents = []
    for doc in documents:
        content = doc.page_content
        content = content.replace('—', ' ')
        for pattern in ['•', '*', '–', '-']:
            content = content.replace(pattern, '')
        doc.page_content = ' '.join(content.split())
        cleaned_documents.append(doc)
    return cleaned_documents

def create_knowledge_base():
    try:
        loader = TextLoader(FILE_PATH, encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        print(f"Error: {FILE_PATH} not found.")
        return

    documents = clean_text_for_embedding(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks = text_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=GPU_SERVER_URL)

    PGVector.from_documents(
        embedding=embeddings,
        documents=chunks,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    print(f"Successfully migrated {len(chunks)} chunks to pgvector on port 5433.")

if __name__ == "__main__":
    create_knowledge_base()
