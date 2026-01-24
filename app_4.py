import streamlit as st
import streamlit_authenticator as stauth
import sqlite3
import pandas as pd
from database import init_db, get_credentials, save_log, add_user
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

init_db()
st.set_page_config(page_title="Gita-CBT Clinical Portal", layout="wide")

# REPLACE THIS with the actual IP address of your GPU machine
GPU_SERVER_URL = "http://192.168.2.221:11434" 

st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="false"] { display: none; width: 0px; }
        [data-testid="stSidebarNav"] { display: none; }
        .main { background-color: #f5f7f9; }
        div[data-testid="stForm"] { background-color: #OOOOOO; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_chain():
    # Utilizing the remote GPU server for LLM and Embeddings
    llm = OllamaLLM(model="deepseek-llm:latest", base_url=GPU_SERVER_URL, temperature=0.6) 
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=GPU_SERVER_URL)
    
    # Points to your local bhagavad_gita_index folder
    vector_store = Chroma(persist_directory="bhagavad_gita_index", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template("""<s>[INST] 
    You are a Clinical Mentor providing structural guidance.
    CURRENT LESSON: {selected_skill}
    
    INSTRUCTIONS:
    1. Use professional empathy.
    2. Provide secular, practical advice based on the provided Context.
    3. Transition into a specific objective for the user to tackle.
    4. NEVER mention religious text names.
    
    Context: {context} \n User: {input} [/INST]</s>""")

    return (RunnablePassthrough.assign(
                context=(lambda x: x['input']) | retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))
            ) | prompt | llm | StrOutputParser())

credentials = get_credentials()
authenticator = stauth.Authenticate(credentials, 'wellbeing_cookie', 'auth_key', cookie_expiry_days=30)

if not st.session_state.get("authentication_status"):
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        authenticator.login()
    with tab2:
        st.subheader("Create a New Account")
        ADMIN_SECRET_KEY = "CLINIC_2024_ADMIN"
        with st.form("registration_form"):
            new_username = st.text_input("Username")
            new_full_name = st.text_input("Full Name")
            new_password = st.text_input("Password", type="password")
            new_password_confirm = st.text_input("Confirm Password", type="password")
            st.divider()
            is_admin = st.checkbox("Register as Administrator") # 
            admin_key = st.text_input("Admin Access Key", type="password")
            if st.form_submit_button("Register"):
                if new_password == new_password_confirm:
                    assigned_role = 'admin' if is_admin and admin_key == ADMIN_SECRET_KEY else 'patient' # 
                    if add_user(new_username, new_password, new_full_name, assigned_role):
                        st.success(f"Registered as {assigned_role}!")
                    else: st.error("User already exists.")

else:
    username = st.session_state.get("username")
    role = credentials['usernames'][username]['role']
    
    with st.sidebar:
        st.title(f"Welcome, {st.session_state.get('name')}")
        authenticator.logout('Logout', 'main')
        st.divider()
        
        # 11-Chapter Clinical Curriculum [cite: 7-17, 18, 21]
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
        selected_skill = st.selectbox("Current Module:", list(patient_curriculum.keys()))
        skill_focus = patient_curriculum[selected_skill]
        
        # Emoji Mood Selector replacing the slider 
        st.write("How are you feeling?")
        mood_map = {"😊": "Happy", "😔": "Low Mood", "😰": "Anxious", "😡": "Angry", "😴": "Tired"}
        cols = st.columns(len(mood_map))
        for i, (emoji, label) in enumerate(mood_map.items()):
            if cols[i].button(emoji):
                st.session_state.current_mood = label
                st.session_state.mood_triggered = True

    if role == "admin":
        st.title("👨‍⚕️ Clinician Admin Dashboard")
        conn = sqlite3.connect('wellbeing.db')
        df = pd.read_sql_query("SELECT * FROM logs", conn)
        conn.close()
        st.dataframe(df, use_container_width=True)

    elif role == "patient":
        st.title("My CBT Skills Portal")
        chat_tab, lesson_tab = st.tabs(["💬 General Chat", "🎓 Structured Lessons"]) # [cite: 20]

        # Bot proactively acknowledges mood [cite: 4]
        if st.session_state.get("mood_triggered"):
            st.session_state.mood_triggered = False
            mood_trigger = f"The user feels {st.session_state.current_mood}. Acknowledge this and start a conversation."
            with st.chat_message("assistant"):
                res_box, full_res = st.empty(), ""
                chain = get_rag_chain()
                for chunk in chain.stream({"input": mood_trigger, "selected_skill": selected_skill, "skill_focus": skill_focus}):
                    full_res += chunk
                    res_box.markdown(full_res + "▌")
                res_box.markdown(full_res)
                save_log(username, st.session_state.current_mood, "[Mood Trigger]", full_res, selected_skill)
                if "messages" not in st.session_state: st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": full_res})

        with chat_tab:
            if "messages" not in st.session_state: st.session_state.messages = []
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])
            if prompt := st.chat_input("How can I help you today?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    res_box, full_res = st.empty(), ""
                    chain = get_rag_chain()
                    for chunk in chain.stream({"input": prompt, "selected_skill": selected_skill, "skill_focus": skill_focus}):
                        full_res += chunk
                        res_box.markdown(full_res + "▌")
                    res_box.markdown(full_res)
                    save_log(username, st.session_state.get("current_mood", "N/A"), prompt, full_res, selected_skill)
                    st.session_state.messages.append({"role": "assistant", "content": full_res})

        with lesson_tab:
            st.header(f"Chapter: {selected_skill}")
            st.info(f"Objective: {skill_focus}") # [cite: 21]
            if st.button("Start 1-Hour Session"): # [cite: 18]
                # Bot takes control of the conversation 
                lesson_prompt = f"Start Chapter {selected_skill}. Lead the session and provide my objectives."
                with st.chat_message("assistant"):
                    res_box, full_res = st.empty(), ""
                    chain = get_rag_chain()
                    for chunk in chain.stream({"input": lesson_prompt, "selected_skill": selected_skill, "skill_focus": skill_focus}):
                        full_res += chunk
                        res_box.markdown(full_res + "▌")
                    res_box.markdown(full_res)