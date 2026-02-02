import streamlit as st
import torch
import os
import multiprocessing
from langchain_community.llms import LlamaCpp 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# --- CONFIGURATION ---
MODEL_FILE = "Mistral-Nemo-12B-Instruct.Q4_K_M.gguf" 
INDEX_PATH = "bhagavad_gita_index"
CHROMA_COLLECTION_NAME = "bhagavad_gita_collection"
EMBEDDING_MODEL_NAME = "nvidia/llama-embed-nemotron-8b"

# Force CPU settings for stability
GPU_DEVICE = "cpu"
CPU_THREADS = multiprocessing.cpu_count()

@st.cache_resource
def get_rag_chain():
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: GGUF model file '{MODEL_FILE}' not found.")
        st.stop()

    # CPU Optimized LLM Initialization
    llm = LlamaCpp(
        model_path=MODEL_FILE,
        n_gpu_layers=0,
        n_threads=CPU_THREADS,
        n_batch=128,
        n_ctx=2048, 
        temperature=0.3,
        max_tokens=1024,
        verbose=False,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}
    )
    
    vector_store = Chroma(
        persist_directory=INDEX_PATH, 
        embedding_function=embeddings, 
        collection_name=CHROMA_COLLECTION_NAME 
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 1. Contextualization: Converts "How do I do that?" into a full search query
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

    # 2. Main Prompt
    system_template = """<s>[INST] 
 You are a wise, secular mentor. Use the provided Context to offer one paragraph of practical, empathetic advice.

CRITICAL RULES:
1. NEVER mention the name of the text, the author, or the philosopher (e.g., Do NOT say 'Based on the Analects' or 'Hippocrates said').
2. NEVER use religious or spiritual terms.
3. Use the Context to provide direct, actionable advice for the user's specific problem.
4. If the user asks multiple questions, integrate the answers into a single, cohesive response.
5. If the response is getting long, prioritize the most impactful advice so you do not cut off mid-sentence.

Context:
{context}

User Input: {input} [/INST]</s>"""

    qa_prompt = ChatPromptTemplate.from_template(system_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Final Chain
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- UI LOGIC ---
st.set_page_config(page_title="Philosophical Analyst", layout="wide")
st.title("Philosophical Analyst of Ancient Texts")

if "messages" not in st.session_state:
    st.session_state.messages = []

rag_chain = get_rag_chain()

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about duty, detachment, or the nature of self..."):
    # Build history (This is where the indentation fix happened)
    history = []
    for m in st.session_state.messages[-5:]: 
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            try:
                ai_response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": history
                })
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")