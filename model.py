from langchain_community.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import chainlit as cl

custom_prompt="""
Use the given context and answer accordingly. If you don't know the answer say so, don't makeup any answers outside the context.
context:{context}

question:{question}

Answer:
"""

def define_prompt_template():
    prompt=PromptTemplate(template=custom_prompt, input_variables=['context', 'question'])
    return prompt

def define_model():
    llm=CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML', model_file='llama-2-7b-chat.ggmlv3.q2_K.bin',model_type='llama', temperature=0.6,max_new_tokens=512)
    return llm

def define_chain(model,prompt,vector_db):
    chain=RetrievalQA.from_chain_type(llm=model,chain_type='stuff',retriever=vector_db.as_retriever(search_kwargs={'k':5}),
                                      return_source_documents=True,chain_type_kwargs={'prompt':prompt})
    return chain

def bot():
    embed=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2',model_kwargs={'device':'cpu'})
    db=FAISS.load_local('vector_database/db_elements',embed)
    llm=define_model()
    prompt=define_prompt_template()
    qa_chain=define_chain(llm,prompt,db)
    return qa_chain

def result(query):
    res=bot()
    answer=res({'query':query})
    return answer


@cl.on_chat_start
async def chat_start():
    qa_bot= bot()
    msg = cl.Message(content='Initializing resources, this might take a minute...')
    await msg.send()
    msg.content='Hello, I am FinanciaLLM. What do you want to know about World Economics?'
    await msg.update()
    cl.user_session.set('QABot', qa_bot)

@cl.on_message
async def mess(message):
    qa_bot = cl.user_session.get('QABot') 
    callback=cl.LangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback.answer_reached = True
    res = await qa_bot.acall(message.content, callbacks=[callback])
    answer = res["result"]
    await cl.Message(content=answer).send()