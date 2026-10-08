import streamlit as st
from langgraph_backend import chatbot , retrive_all_threads
from langchain_core.messages import HumanMessage
import uuid

st.set_page_config(page_title='LangGraph Chatbot', page_icon='🤖')

# ************************** Utility Functions **************************************************************

def generate_thread_id() :

    thread_id = str(uuid.uuid4())
    return thread_id

def reset_chat() :
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id) : 
    if thread_id not in st.session_state['chat_threads'] :
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    return chatbot.get_state(config={'configurable':{'thread_id': thread_id}}).values.get('messages',[])

def get_title(thread_id):
    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage):
            return msg.content[:30]
    return 'New Chat'

# st.session_state -> dict ->  no data loss

# ******************************** Session SetUp***********************************************************

if 'message_history' not in st.session_state : 
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state :

    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:

    st.session_state['chat_threads'] = retrive_all_threads()

add_thread(st.session_state['thread_id'])


# ******************************** SideBar UI **************************************************************

st.sidebar.title('👾 LangGraph Chatbot')

if st.sidebar.button('➕ New Chat', use_container_width=True, type='primary') : 
    reset_chat()

st.sidebar.divider()
st.sidebar.caption('RECENT CHATS')

for thread_id in reversed(st.session_state['chat_threads']) : 
    title = get_title(thread_id)
    if title == 'New Chat' :
        continue
    if st.sidebar.button(title, key=f'thread-{thread_id}', use_container_width=True) : 
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id) 

        temp_messages = []

        for msg in messages:
            if isinstance(msg,HumanMessage): 
                role = 'user'

            else :
                role = 'assistant'
            temp_messages.append({'role' : role , 'content' : msg.content})

        st.session_state['message_history'] = temp_messages


# ************************************** Main UI *************************************************************

CONFIG = {'configurable':{'thread_id':st.session_state['thread_id']}}
user_input = st.chat_input('Ask me anything...')

if not st.session_state['message_history'] and not user_input :
    st.markdown('### 👋 Hi! What would you like to talk about?')
    st.caption('Powered by LangGraph + Groq')

# loading conversation History
for message in st.session_state['message_history'] :
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if user_input :

    # add the message first to the message_history
    
    st.session_state['message_history'].append({'role':'user','content':user_input})

    with st.chat_message('user') :
        st.markdown(user_input)


    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk , metadata in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role':'assistant','content':ai_message})