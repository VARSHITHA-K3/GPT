import os
from langchain import LlamaCpp
import streamlit as st
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from huggingface_hub import hf_hub_download
from langchain.llms import GPT4All
from transformers import LlamaTokenizer, LlamaForCausalLM
from constants import MODEL_ID,MODEL_BASENAME, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY

#load_dotenv()
embeddings_model_name = EMBEDDING_MODEL_NAME
persist_directory = PERSIST_DIRECTORY
model_id = MODEL_ID
model_basename = MODEL_BASENAME
#target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))


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

def load_model(model_id, model_basename):
        if ".ggml" in model_basename:
            #logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 2048
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            return LlamaCpp(**kwargs)
        else:
            #logging.info("Using LlamaTokenizer")
            tokenizer = LlamaTokenizer.from_pretrained(model_id)
            model = LlamaForCausalLM.from_pretrained(model_id)

def handle_userinput(user_question):
    
    vectorstore = get_vectorstore()
    if "conversation" not in st.session_state:
        response, sources= st.session_state.conversation = get_conversation_chain(vectorstore)
        return response, sources
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        

def main():
    load_dotenv()
    st.set_page_config(page_title="EmilyGPT v2.0",
                       page_icon=":sunglasses:")
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.write(css, unsafe_allow_html=True)
    # with st.sidebar:
    #     st.subheader("Your documents")
    vectorstore = get_vectorstore()
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversation_chain(vectorstore)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    #st.header("EmilyGPT v2.0 🤖")
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


if __name__ == '__main__':
    main()