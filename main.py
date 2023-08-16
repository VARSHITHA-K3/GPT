from fastapi import FastAPI
from typing import List
from app import get_conversation_chain,get_vectorstore
app = FastAPI()

vectorstore = get_vectorstore()
conversation_chain = get_conversation_chain(vectorstore)
chat_history = []

@app.post("/ask_question/")
async def ask_question(req_info: dict):
    #handle_userinput(req_info['question'])
    user_question = req_info.get('question')
    response = conversation_chain({'question': user_question})
    chat_history.append({'user': user_question, 'bot': response['answer']})
    return {"bot_response": response['answer']}

@app.post("/ask_question1/")
async def ask_question(req_info: dict):
    #handle_userinput(req_info['question'])
    user_question = req_info.get('question')
    response = conversation_chain({'question': user_question})
    chat_history.append({'user': user_question, 'bot': response['answer']})
    return {"bot_response": response['answer']}

@app.get("/chat_history/")
async def get_chat_history():
    return {"chat_history": chat_history}
