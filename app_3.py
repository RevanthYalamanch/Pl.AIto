import streamlit as st
import torch
import os
import multiprocessing
from langchain_community.llms import LlamaCpp 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

# --- CPU OPTIMIZATION ---
MODEL_FILE = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
INDEX_PATH = "bhagavad_gita_index"
CHROMA_COLLECTION_NAME = "bhagavad_gita_collection"
EMBEDDING_MODEL_NAME = "nvidia/llama-embed-nemotron-8b"
CPU_THREADS = multiprocessing.cpu_count()

@st.cache_resource
def get_rag_chain():
    # Lowered n_ctx and max_tokens for faster CPU response
    llm = LlamaCpp(
        model_path=MODEL_FILE,
        n_gpu_layers=0,
        n_threads=CPU_THREADS,
        n_batch=8, 
        n_ctx=1024, 
        temperature=0.2,
        max_tokens=350, 
        streaming=True, 
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Main Patient-Facing Prompt
    system_template = """<s>[INST] 
You are an application designed to help mental help professionals with treating their patients. The app follows an eleven step plan, which 
is listed in the patient_curriculum list below.

Your main goals are to analyze the user's state, provide a reassuring perspective, and to challenge the user with a piece of self-reflection. 

Make sure to speak naturally, acknowledge the user's struggle, and include the perspective from the provided bhagavad_gita_index. 


After acknowledging the problem, transition into a specific, practical step that the user can take righ now to manage their life.

**MAKE SURE TO NOT REFERENCE ANY OF THE RELIGIOUS OR PHILOSOPHICAL TEXTS, AND INSTEAD TO FOCUS ON PROVIDING REAL, PRACTICAL ADVICE THAT CAN APPLY 
TO ANYBODY FROM DIFFERENT WALKS OF LIFE.**

Context:
{context}

User Input: {input} [/INST]</s>"""

    qa_prompt = ChatPromptTemplate.from_template(system_template)
    
    # We remove the contextualization chain for now to increase CPU SPEED
    # Direct retrieval based on raw input is faster on limited hardware
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: x['input']) | retriever | format_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- PATIENT SKILLS CURRICULUM ---
patient_curriculum = {
    "1. Introduction": "Understanding how we can work together to improve your wellbeing.",
    "2. Understanding Low Mood & Anxiety": "Identifying the cycles of panic, worry, and sadness.",
    "3. The ABC Model": "Learning how Situations (A) lead to Beliefs (B) and Consequences (C).",
    "4. Setting SMART Goals": "Making small, specific plans for recovery.",
    "5. Lifestyle Changes": "Improving sleep, diet, and daily routine for mental health.",
    "6. Increasing Activity Levels": "Using Behavioral Activation to fight depression.",
    "7. Facing Your Fears": "Techniques for overcoming anxiety and panic in the moment.",
    "8. Containing Worry": "Setting 'Worry Time' and managing circular thoughts.",
    "9. Problem Solving": "Breaking down overwhelming obstacles into small steps.",
    "10. Thought Challenging": "Cognitive Restructuring to change harmful thought patterns.",
    "11. Wellbeing Blueprint": "Creating your long-term plan for staying well."
}

# --- UI LOGIC ---
st.set_page_config(page_title="CBT Skills Mentor", layout="wide")

with st.sidebar:
    st.title("📂 My Skills Workbook")
    selected_skill = st.selectbox("Current Module:", list(patient_curriculum.keys()))
    skill_focus = patient_curriculum[selected_skill]
    st.info(f"**Focus:** {skill_focus}")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("CBT Skills Training Support")
st.caption(f"Currently focusing on: {selected_skill}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process Input
if prompt := st.chat_input("How are you feeling right now?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # STREAMING for speed and "instant" feel
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            rag_chain = get_rag_chain()
            for chunk in rag_chain.stream({
                "input": prompt,
                "selected_skill": selected_skill,
                "skill_focus": skill_focus
            }):
                full_response += chunk
                response_placeholder.markdown(full_response + "")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {e}")