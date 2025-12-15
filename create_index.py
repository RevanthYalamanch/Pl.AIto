import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

FILE_PATH = "scripture_text.txt"
INDEX_PATH = "bhagavad_gita_index"
EMBEDDING_MODEL_NAME = "nvidia/llama-embed-nemotron-8b"
CHROMA_COLLECTION_NAME = "bhagavad_gita_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250

def clean_text_for_embedding(documents):
    cleaned_documents = []
    em_dash_pattern = '—'
    bullet_patterns = ['•', '*', '–', '-']
    
    for doc in documents:
        content = doc.page_content
        
        content = content.replace(em_dash_pattern, ' ')
        
        for pattern in bullet_patterns:
            content = content.replace(pattern, '')
            
        content = ' '.join(content.split())
        
        doc.page_content = content
        cleaned_documents.append(doc)
    return cleaned_documents

def create_knowledge_base():
    if os.path.exists(INDEX_PATH):
        return

    try:
        loader = TextLoader(FILE_PATH, encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        print(f"Error: File '{FILE_PATH}' not found. Please create it and paste your scripture text inside.")
        return

    documents = clean_text_for_embedding(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True} 
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
