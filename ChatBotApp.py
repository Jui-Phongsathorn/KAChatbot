#import libralies
import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

# App title Setting
st.set_page_config(page_title="🤖💬 KrungsriAuto Guru Chatbot")

# Create Interactive Sidebar
with st.sidebar:
    st.title('🤖💬 KrungsriAuto Guru Chatbot')
    st.write('This chatbot is created using the GPT LLM model from OpenAI.')
    openai_api = st.text_input('Enter OpenAI API token:', type='password')
    if openai_api =='juijuizze':
        f = open(r".\openai_api_key.txt")
        openai_api = f.read()    
        st.success('Welcome Jui 😊')
    elif not (openai_api.startswith('sk_') and len(openai_api)==51):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['openai_api'] = openai_api

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a LLM model', ['gpt-3.5-turbo','gpt-3.5-turbo-1106', 'gpt-4'], key='selected_model')
    if selected_model == 'gpt-3.5-turbo':
        gpt_model = 'gpt-3.5-turbo'
    elif selected_model == 'gpt-3.5-turbo-1106':
        gpt_model = 'gpt-3.5-turbo-1106'
    elif selected_model ==  'gpt-4':
        gpt_model = 'gpt-4'
    
    st.subheader('Objective')
    purpose = st.radio('who are you looking for?',["C4C guru", "New Car/MC guru"])

    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    st.markdown('📖 Learn more about how to create UI using streamlit in this [blog](https://streamlit.io/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ วันนี้มีอะไรให้กระผมรับใช้ครับ 🤗?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "สวัสดีครับ วันนี้มีอะไรให้กระผมรับใช้ครับ 🤗?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Setup LLM object to use and embedding function (same function as used in vectorstore) 
llm = ChatOpenAI(temperature=temperature, model_name= gpt_model, openai_api_key = openai_api)
embedding_func = OpenAIEmbeddings(api_key= openai_api)

# We separated knowledge base for two chatbots
if purpose == 'C4C guru':
    db = Chroma(persist_directory = './DataSource/article_info_knowlege_from_scraping.db', embedding_function= embedding_func)
elif purpose == 'New Car/MC guru':
    db = Chroma(persist_directory = './DataSource/car_price_knowlege_from_scraping.db', embedding_function= embedding_func)

retriever = db.as_retriever()

memory = ConversationBufferMemory(memory_key="chat_history", input_key = 'question',return_messages=True)

# Set up the prompt template
general_system_template = r""" 
----
{context}
----
Your answer must not begin with 'ANSWER:' just tell the content of an answer 
"""
general_user_template ="```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )

conversationchain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                combine_docs_chain_kwargs={'prompt' : qa_prompt},
                                                verbose=True,
                                                retriever= retriever,
                                                memory=memory)

context_setup = 'You are a friendly sales employee working for krungsri auto (in Thai: กรุงศรีออโต้) who answer what customer asks concisely (in Thai) based on information you got from the retriever. If you do not know just said you do not know\
    ข้อมูลเพิ่มเติม: "Car4Cash", "C4C" และ "คาร์ ฟอร์ แคช" คือสิ่งเดียวกัน.'

# Function to return GPT response
def generate_gpt_response(query):
    output = conversationchain({'question': query,'context': context_setup})
    return output['answer']

# User-provided prompt
if prompt := st.chat_input(disabled=not openai_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            response = generate_gpt_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


