import streamlit as st
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# Core LangChain modular imports
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION AND MODEL DEFINITIONS ---
MODEL_ID = "microsoft/phi-2"
INDEX_PATH = "bhagavad_gita_index"
CHROMA_COLLECTION_NAME = "bhagavad_gita_collection"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GPU_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def get_rag_chain():
    """Initializes and returns the complete RAG chain and the retriever."""

    # --- A. Load Components (Quantized LLM and Tokenizer) ---
    st.write("Initializing Mistral 7B (4-bit quantization)...")
    
    # 1. Define 4-bit Quantization Config (CRITICAL for 16GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 if GPU_DEVICE == "cuda" else torch.float32,
        # FIX for meta tensor error: Explicitly allow offloading to CPU RAM
        llm_int8_enable_fp32_cpu_offload=True 
    )

    # 2. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Let accelerate manage device placement
        low_cpu_mem_usage=True # Hint for lower system RAM usage
    )

    # 3. Create HuggingFace Pipeline (Inference LLM)
    hf_pipeline = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.1,
        return_full_text=False,
        batch_size=4,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id

    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # --- B. Load Retriever (Your ChromaDB Index) ---
    st.write("Loading ChromaDB Knowledge Base...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': GPU_DEVICE}
    )
    
    if not os.path.exists(INDEX_PATH):
        st.error(f"Error: Index path '{INDEX_PATH}' not found. Did you run create_index.py successfully?")
        st.stop()
        
    # Load ChromaDB instance
    vector_store = Chroma(
        persist_directory=INDEX_PATH, 
        embedding_function=embeddings, 
        collection_name=CHROMA_COLLECTION_NAME 
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks

    # --- C. Define Prompt and Chain (Pure LCEL Construction) ---

    def format_docs(docs):
        # Helper function to concatenate documents into a single string for the prompt
        return "\n\n".join(doc.page_content for doc in docs)
    
    system_template = """
You are an AI Philosophical Analyst and Expert Text Interpreter. Your purpose is to analyze and explain the core philosophical, ethical, and psychological concepts (such as duty, action, self, and consequence) found within the ancient texts provided below.
Your responses must be strictly secular, focusing on the practical philosophy and universal human experience reflected in the text. Avoid using overtly religious language or doctrine (e.g., replace 'Lord Krishna' with 'The speaker' or 'The teacher').
Your answer must be based **ONLY** on the provided context. If the context is insufficient to answer the user's question, state clearly, 'I cannot find a direct philosophical analysis of that concept in the provided text, but I can offer related material.'
Context:
{context}
"""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])
    
    # 1. Define the Answer Generator chain
    answer_chain = (
        rag_prompt 
        | llm 
        | StrOutputParser()
    )

    # 2. Define the chain that retrieves context and assigns the final answer (Structured Output)
    # This structure returns the dictionary: {'answer': str, 'context': list[Document]}
    rag_chain = RunnablePassthrough.assign(
        # context: gets the raw list of Documents for later display
        context=(lambda x: x['input']) | retriever,
    ).assign(
        # answer: assigns the final string answer after processing context through the prompt/LLM
        answer=(
            # This sub-chain formats the context and passes it to the answer_chain
            RunnablePassthrough.assign(context=(lambda x: x['input']) | retriever | RunnableLambda(format_docs))
            | answer_chain
        )
    )
    
    st.success("System Ready: Mistral 7B and Scripture Index Loaded.")
    
    # Return both the LCEL chain and the retriever
    return rag_chain, retriever 

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Spiritual Guidance LLM", layout="wide")
st.title("Spiritual Guidance Helper")
st.markdown("Ask any question about the scriptures, and the LLM will answer based **only** on the indexed texts.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Load the RAG chain and retriever (Unpacks the objects from the cached function)
# The spinner is added here so the UI shows the loading state while the model loads
with st.spinner("Initial setup: Loading the LLM and vector index. This may take a minute or two..."):
    try:
        rag_chain, retriever = get_rag_chain()
    except Exception as e:
        # If loading fails, display the error and stop the app
        st.error(f"Failed to load RAG components. Please check terminal for errors. Last Error: {e}")
        st.stop()
    
# --- 3. CHAT INTERFACE LOGIC ---

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask any question..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate the RAG response
    with st.spinner("Consulting the ancient texts..."):
        try:
            # 1. Invoke the RAG chain (Output is a dict: {'answer': str, 'context': list[Document]})
            response = rag_chain.invoke({"input": prompt}) 
            
            # 2. Extract the answer and context
            ai_response = response['answer']
                           
            final_response = ai_response

        except Exception as e:
            # This catches any errors during generation and displays them gracefully
            final_response = f"An error occurred during generation: {e}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(final_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": final_response})