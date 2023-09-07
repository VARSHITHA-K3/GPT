import logging
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
from constants import (MODEL_ID,MODEL_BASENAME, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY,MODEL_PATH,MODEL_TYPE,
                       TARGET_SOURCE_CHUNKS,MODEL_N_CTX,MODEL_N_BATCH)

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
    "user1": "password1",
    "user2": "password2",
}


def generate_token(username):
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=0,minutes=30),
        'iat': datetime.datetime.utcnow()
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    print (token)
    return token


def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail='Signature has expired')
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Invalid token')


def LoggedIn_Clicked(username, password):
    if username in USERS and USERS[username] == password:
        #st.success("Logged in successfully!")
        token = generate_token(username)
        print(token)
        st.session_state['token'] = token
        st.session_state['username'] = username
        return
    else:
        st.session_state['token'] = False
        st.session_state['username'] = None
        st.error("Invalid user name or password.")

# def LoggedIn_url(username, password):
#     if st.experimental_get_query_params()['username'][0] in USERS and password == 'pass1':
#         print("a")
#         token = generate_token(username)
#         st.session_state['token'] = token
#         st.session_state['username'] = username
#         print(st.session_state,token,username)
#         return token
#     else:
#         st.session_state['token'] = False
#         st.session_state['username'] = None
#         st.error("Invalid user name or password..")
        
def login():
    st.title("Login")
    
    if st.session_state['token'] == False:
        print("A1")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        print("A2")
        st.button ("Login", on_click=LoggedIn_Clicked, args= (username, password))
        print("A3")
    else:
        return ("Invalid")
    query_params = st.experimental_get_query_params()
    if 'username' in query_params and query_params['username'][0] in USERS:
        print("A4")
        username = st.experimental_get_query_params()['username']
        password = USERS['user1']
        if st.experimental_get_query_params()['username'][0] in USERS and USERS['user1'] == password:
            print("A5")
            token = generate_token(username)
            st.session_state['token'] = True
            st.session_state['username'] = True
            print(st.session_state,token,username)
            print("A6")
        else:
            st.session_state['token'] = False
            st.session_state['username'] = False
            st.error("Invalid user name or password..")
    else:
        return ("invalid")

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    if "db" not in st.session_state:
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=None)
        st.session_state.db = db
        vectorstore = db.as_retriever()
        return vectorstore

def get_conversation_chain(vectorstore):
    llm = load_model()
    memory = ConversationBufferMemory(input_key='question',memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore,
        memory=memory
    )
    return conversation_chain



def load_model():
        if MODEL_TYPE=="Llama":
            max_ctx_size = 2048
            return LlamaCpp(model_path=model_path,n_ctx = max_ctx_size,max_tokens = max_ctx_size)
        elif MODEL_TYPE=="GPT":
            return GPT4All(model=model_path, backend='gptj', n_batch=model_n_batch,verbose=None)
        else:
            return None
    

def handle_userinput(user_question,vectorstore:any):
    
    vectorstore = get_vectorstore()
    if "conversation" not in st.session_state:
        response, sources= st.session_state.conversation = get_conversation_chain(vectorstore)
        return response, sources
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

def ask_question(vectorstore:any):
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(user_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace(
                        "{{MSG}}", message.content), unsafe_allow_html=True)
        with st.expander("Show Source Documents"):
            search = st.session_state.db.similarity_search_with_score(user_question)
            for i, doc in enumerate(search): 
                # print(doc)
                st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
                st.write(doc[0].page_content) 
                st.write("--------------------------------")

def get_history(vectorstore:any):
    if "conversation" not in st.session_state or "chat_history" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.chat_history = None


def main():
    st.write(css, unsafe_allow_html=True)
    vectorstore = get_vectorstore()
    get_history(vectorstore)
    ask_question(vectorstore)

def headerSection():
    st.write(css, unsafe_allow_html=True)
    if 'token' not in st.session_state and 'username' not in st.session_state:
        st.session_state['token'] = False
        st.session_state['username'] = None
        login() 
    else:
        if st.session_state['token'] and st.session_state['username']:
            main()
        else:
            login()