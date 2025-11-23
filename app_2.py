import streamlit as st
import torch
import os
from langchain_community.llms import LlamaCpp 
from langchain_core.prompts import PromptTemplate 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf" 
INDEX_PATH = "bhagavad_gita_index"
CHROMA_COLLECTION_NAME = "bhagavad_gita_collection"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_rag_chain():
    st.write(f"Initializing LlamaCpp Model: {MODEL_FILE}...")
    
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: GGUF model file '{MODEL_FILE}' not found.")
        st.stop()

    llm = LlamaCpp(
        model_path=MODEL_FILE,
        n_gpu_layers=-1 if GPU_DEVICE == "cuda" else 0,
        n_batch=512,
        n_ctx=4096, 
        temperature=0.1,
        max_tokens=512,
        verbose=False,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': GPU_DEVICE}
    )
    
    if not os.path.exists(INDEX_PATH):
        st.error(f"Error: Index path '{INDEX_PATH}' not found.")
        st.stop()

    vector_store = Chroma(
        persist_directory=INDEX_PATH, 
        embedding_function=embeddings, 
        collection_name=CHROMA_COLLECTION_NAME 
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    system_template = """
    You are an AI program designed to help mental health professionals by providing insights based on ancient philosophical texts. You are to
    answer questions from user providing guidance strictly based on the content provided below. You are *NOT* to mention any religious or  
    spiritual aspects, but rather focus on practical philosophy and universal human experience. 

Context:
{context}
"""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])
    
    answer_chain = (
        rag_prompt 
        | llm 
        | StrOutputParser()
    )

    rag_chain = RunnablePassthrough.assign(
        context=(lambda x: x['input']) | retriever,
    ).assign(
        answer=(
            RunnablePassthrough.assign(context=(lambda x: x['input']) | retriever | RunnableLambda(format_docs))
            | answer_chain
        )
    )
    
    st.success(f"System Ready: {MODEL_FILE} Loaded via LlamaCpp.")
    
    return rag_chain, retriever 


st.set_page_config(page_title="Bhagavad Gita Philosophical Analyst", layout="wide")
st.title("Philosophical Analyst of Ancient Texts")
st.markdown("Ask any question about the ethical or philosophical principles in the texts.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
with st.spinner("Initial setup: Loading the LlamaCpp Model and vector index. This may take a minute..."):
    try:
        rag_chain, retriever = get_rag_chain()
    except Exception as e:

        st.error(f"Failed to load RAG components. Please check terminal for errors. Last Error: {e}")
        st.stop()
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about duty, detachment, or the nature of self..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing the text for philosophical context..."):
        try:
            response = rag_chain.invoke({"input": prompt}) 
            ai_response = response['answer']
            retrieved_docs = response['context']
            source_content = "\n\n---\n*Analysis complete.*"
            final_response = ai_response + source_content

        except Exception as e:
            final_response = f"An error occurred during generation: {e}"


    with st.chat_message("assistant"):
        st.markdown(final_response)

    st.session_state.messages.append({"role": "assistant", "content": final_response})