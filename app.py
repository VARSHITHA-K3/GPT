import os
from fastapi import HTTPException
from langchain.llms import LlamaCpp
import streamlit as st
import jwt
import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import GPT4All
from constants import MODEL_ID,MODEL_BASENAME, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY,MODEL_PATH,MODEL_TYPE,TARGET_SOURCE_CHUNKS,MODEL_N_CTX,MODEL_N_BATCH

#load_dotenv()
embeddings_model_name = EMBEDDING_MODEL_NAME
persist_directory = PERSIST_DIRECTORY
model_id = MODEL_ID
model_basename = MODEL_BASENAME
model_path = MODEL_PATH
target_source_chunks = TARGET_SOURCE_CHUNKS
model_n_ctx = MODEL_N_CTX
model_n_batch = MODEL_N_BATCH

SECRET_KEY = "SECRET"
USERS = {
    "user1": "password1"
}

def login_page():
    #st.title("Login")
    # auto_login_username = "user1"
    # auto_login_password = USERS.get(auto_login_username)
    
    # if auto_login_password:
    #     #st.success(f"Auto-logged in as {auto_login_username}!")
    #     token = generate_token(auto_login_username)
    #     st.experimental_set_query_params(token=token)
    #     st.experimental_rerun()
    
    st.warning("Please use the login form.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.success("Logged in successfully!")
            token = generate_token(username)
            st.experimental_set_query_params(token=token)
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

def generate_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=0,minutes=30)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    print (token)
    return token

def verify_token(token):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    payload = verify_token(token)
    if payload:
        st.session_state.logged_in = True
        st.success(f"Logged in as: {payload['username']}")
    else:
        st.error("Invalid or expired token")
    return payload


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    if "db" not in st.session_state:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=None)
        st.session_state.db = db
        vectorstore = db.as_retriever()
        #vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

def get_conversation_chain(vectorstore):
    llm = load_model(model_id=MODEL_ID, model_basename=MODEL_BASENAME)
    memory = ConversationBufferMemory(input_key='question',memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore,
        memory=memory
    )
    return conversation_chain

def load_model(model_id,model_basename=None):
        if MODEL_TYPE=="Llama":
            #logging.info("Using Llamacpp for GGML quantized models")
            #model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            return LlamaCpp(model_path=model_path,n_ctx = max_ctx_size,max_tokens = max_ctx_size)
        elif MODEL_TYPE=="GPT":
            return GPT4All(model=model_path, backend='gptj', n_batch=model_n_batch,verbose=None)
        else:
            return None
    

def handle_userinput(user_question):
    
    vectorstore = get_vectorstore()
    if "conversation" not in st.session_state:
        response, sources= st.session_state.conversation = get_conversation_chain(vectorstore)
        return response, sources
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        

def main():
    st.set_page_config(page_title="EmilyGPT v2.0",page_icon=":sunglasses:")
    #st.sidebar.write(f"Logged in as: {payload['username']}")
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.write(css, unsafe_allow_html=True)

    username = st.experimental_get_query_params().get("username", [None])[0]
    password = st.experimental_get_query_params().get("password", [None])[0]

    if username in USERS and USERS[username] == password:
        st.session_state.logged_in = True
        token = generate_token(username)
        st.experimental_set_query_params(token=token)
    token = st.experimental_get_query_params().get("token", [None])[0]
    if token:
        vectorstore = get_vectorstore()
        if "conversation" not in st.session_state:
            st.session_state.conversation = get_conversation_chain(vectorstore)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        user_question = st.text_input("Ask a question about your documents:")
    
        if user_question:
            handle_userinput(user_question)
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            with st.expander("Show Source Documents"):
                search = st.session_state.db.similarity_search_with_score(user_question)
                for i, doc in enumerate(search):
                    st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
                    st.write(doc[0].page_content)
                    st.write("--------------------------------")
    else:
        login_page()
    

if __name__ == '__main__':
    main()