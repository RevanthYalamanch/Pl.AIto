import streamlit as st
import streamlit_authenticator as stauth
import pandas as pd
from database import init_db, get_credentials, save_log, add_user
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_postgres import PGVector 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# --- Initial Configuration ---
init_db()
st.set_page_config(page_title="Gita-CBT Clinical Portal", layout="wide")


GPU_SERVER_URL = config['networks']['gpu_server']
LLM_MODEL = config['models']['chat']
PG_CONN_STRING = f"postgresql+psycopg2://postgres:acakdoir@{config['networks']['db_host']}:{config['networks']['db_port']}/{config['networks']['db_name']}"

# --- Global Curriculum ---
curriculum = {
    "1. Introduction": "Orienting to the Gita-CBT framework and setting expectations.",
    "2. Low Mood & Anxiety": "Identifying cycles of thoughts and feelings (Gita 2.14).",
    "3. The ABC Model": "Connecting Situations, Beliefs, and Consequences (Gita 2.62).",
    "4. SMART Goals": "Values-based goal setting without attachment to results (Gita 2.47).",
    "5. Lifestyle Changes": "Sattvic living: Balance in sleep, diet, and work (Gita 6.17).",
    "6. Increasing Activity": "Behavioral activation through selfless action (Karma Yoga).",
    "7. Facing Your Fears": "Exposure therapy and finding inner courage (Gita 2.3).",
    "8. Containing Worry": "Developing focus and managing the 'turbulent' mind (Gita 6.34).",
    "9. Problem Solving": "Discrimination (Buddhi) to break obstacles into steps (Gita 18.30).",
    "10. Thought Challenging": "Cognitive restructuring: Seeing truth vs. delusion (Gita 18.20).",
    "11. Wellbeing Blueprint": "Consolidating practice for long-term equanimity (Gita 2.56)."
}

# --- Custom CSS ---
st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="false"] { display: none; width: 0px; }
        [data-testid="stSidebarNav"] { display: none; }
        .stButton>button { width: 100%; font-size: 24px; border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# --- RAG Setup ---
@st.cache_resource
def get_rag_chain(mode="general"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=GPU_SERVER_URL)
    llm = OllamaLLM(model="deepseek-r1", base_url=GPU_SERVER_URL, temperature=0.4)
    
    vectorstore = PGVector(
        connection=PG_CONN_STRING,
        collection_name="bhagavad_gita_collection",
        embeddings=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
    You are a {role}. 
    Context: {context} 
    User: {input}
    """
    if mode == "lesson":
        role_desc = "STRICT Clinical Instructor for the 11-Hour Course. Focus on: {selected_skill}"
    else:
        role_desc = "supportive Gita-inspired Companion focused on empathy."

    prompt = ChatPromptTemplate.from_template(template.replace("{role}", role_desc))

    return (RunnablePassthrough.assign(
                context=(lambda x: x['input']) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))
            ) | prompt | llm | StrOutputParser())

# --- Authentication Logic ---
credentials = get_credentials()
authenticator = stauth.Authenticate(
    credentials=credentials,
    cookie_name='gita_mentor_v1',
    cookie_key='qwerty312', 
    cookie_expiry_days=30,
    auto_hash=False
)

if not st.session_state.get("authentication_status"):
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        # authenticator.login() now returns the status and username directly in newer versions
        authenticator.login()
    with tab2:
        st.subheader("Create a New Account")
        ADMIN_SECRET_KEY = "CLINIC_2024_ADMIN" 
        with st.form("registration_form"):
            new_username = st.text_input("Username")
            new_full_name = st.text_input("Full Name")
            new_pw = st.text_input("Password", type="password")
            is_admin = st.checkbox("Register as Administrator")
            admin_key = st.text_input("Admin Access Key (If applicable)", type="password")
            
            if st.form_submit_button("Register"):
                role_to_assign = 'admin' if is_admin and admin_key == ADMIN_SECRET_KEY else 'patient'
                if add_user(new_username, new_pw, new_full_name, role_to_assign):
                    st.success(f"Registered as {role_to_assign}!")
                else: st.error("Registration failed.")

else:
    # --- Authenticated View ---
    username = st.session_state.get("username")
    role = credentials['usernames'][username]['roles'][0]
    
    with st.sidebar:
        st.title(f"Hi, {st.session_state.get('name')}")
        authenticator.logout('Logout', 'main')

    if role == "admin":
        st.title("👨‍⚕️ Clinician Admin Dashboard")
        conn = sqlite3.connect('wellbeing.db')
        df = pd.read_sql_query("SELECT * FROM logs", conn)
        conn.close()
        st.dataframe(df, use_container_width=True)

    elif role == "patient":
        if "messages" not in st.session_state: st.session_state.messages = []
        if "lesson_messages" not in st.session_state: st.session_state.lesson_messages = []

        st.title("Gita Clinical Mentor")
        st.markdown("### How are you feeling right now?")
        
        mood_map = {"😊": "Happy", "😔": "Low Mood", "😰": "Anxious", "😡": "Angry", "😴": "Tired"}
        mood_cols = st.columns(len(mood_map))
        
        for i, (emoji, label) in enumerate(mood_map.items()):
            if mood_cols[i].button(emoji, key=f"mood_{i}"):
                st.session_state.messages.append({"role": "user", "content": f"I feel {label}."})
                st.session_state.trigger_response = True 

        st.divider()
        chat_tab, lesson_tab = st.tabs(["💬 General Chat", "🎓 Structured Lessons"])

        with chat_tab:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])

            if st.session_state.get("trigger_response"):
                st.session_state.trigger_response = False
                with st.chat_message("assistant"):
                    chain = get_rag_chain(mode="general")
                    res = st.write_stream(chain.stream({"input": st.session_state.messages[-1]["content"]}))
                    st.session_state.messages.append({"role": "assistant", "content": res})

            if prompt := st.chat_input("Type to chat..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    chain = get_rag_chain(mode="general")
                    res = st.write_stream(chain.stream({"input": prompt}))
                    st.session_state.messages.append({"role": "assistant", "content": res})

        with lesson_tab:
            with st.container(border=True):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric("Hour", current_hour)
                with col2:
                    st.markdown(f"### {lesson_data['title']}")
                    st.progress(current_hour / 11)
                
                st.markdown(f"**Ancient Hook:** Gita Verse {lesson_data['verse']}")
                st.markdown(f"**Clinical Tool:** {lesson_data['focus']}")
                selected_skill = st.selectbox("Current Module:", list(curriculum.keys()))
                st.info(f"**Objective:** {curriculum[selected_skill]}")
                
                for m in st.session_state.lesson_messages:
                    with st.chat_message(m["role"]): st.markdown(m["content"])

                if st.button("Start 1-Hour Session"):
                    with st.chat_message("assistant"):
                        chain = get_rag_chain(mode="lesson")
                        res = st.write_stream(chain.stream({"input": "Start lesson", "selected_skill": selected_skill, "skill_focus": curriculum[selected_skill]}))
                        st.session_state.lesson_messages.append({"role": "assistant", "content": res})

                if l_prompt := st.chat_input("Speak to Instructor...", key="l_input"):
                    st.session_state.lesson_messages.append({"role": "user", "content": l_prompt})
                    with st.chat_message("user"): st.markdown(l_prompt)
                    with st.chat_message("assistant"):
                        chain = get_rag_chain(mode="lesson")
                        res = st.write_stream(chain.stream({"input": l_prompt, "selected_skill": selected_skill, "skill_focus": curriculum[selected_skill]}))
                        st.session_state.lesson_messages.append({"role": "assistant", "content": res})