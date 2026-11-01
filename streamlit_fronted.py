import streamlit as st
from langgraph_backend import chatbot , retrive_all_threads , ingest_pdf , thread_document_metadata   # NEW (added ingest_pdf, thread_document_metadata)
from langchain_core.messages import HumanMessage , AIMessage , ToolMessage
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

if 'ingested_docs' not in st.session_state :                                  
    st.session_state['ingested_docs'] = {}                                    

add_thread(st.session_state['thread_id'])

thread_key = str(st.session_state['thread_id'])                               
thread_docs = st.session_state['ingested_docs'].setdefault(thread_key, {})    


# ******************************** SideBar UI **************************************************************

st.sidebar.title('👾 LangGraph Chatbot')

if st.sidebar.button('➕ New Chat', use_container_width=True, type='primary') : 
    reset_chat()

# ---- PDF upload (NEW) ----
if thread_docs :                                                             
    latest_doc = list(thread_docs.values())[-1]                               
    st.sidebar.success(                                                    
        f"Using `{latest_doc.get('filename')}` "                              
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)" 
    )                                                                         
else :                                                                        
    st.sidebar.info('No PDF indexed yet.')                                    

uploaded_pdf = st.sidebar.file_uploader(                                      
    'Upload a PDF for this chat', type=['pdf'], key=f'pdf-{thread_key}'       
)                                                                             
if uploaded_pdf :                                                          
    if uploaded_pdf.name in thread_docs :                                     
        st.sidebar.info(f'`{uploaded_pdf.name}` already processed for this chat.')  
    else :                                                                    
        with st.sidebar.status('Indexing PDF…', expanded=True) as status_box :  
            summary = ingest_pdf(                                            
                uploaded_pdf.getvalue(),                                     
                thread_id=thread_key,                                        
                filename=uploaded_pdf.name,                                   
            )                                                                 
            thread_docs[uploaded_pdf.name] = summary                         
            status_box.update(label='✅ PDF indexed', state='complete', expanded=False) 
# ---- end PDF upload ----

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

            elif isinstance(msg,AIMessage) and msg.content :
                role = 'assistant'

            else :
                continue
            temp_messages.append({'role' : role , 'content' : msg.content})

        st.session_state['message_history'] = temp_messages


# ************************************** Main UI *************************************************************

CONFIG = {
    'configurable':{'thread_id':st.session_state['thread_id']},
          "metadata":{"thread_id":st.session_state["thread_id"]},
          'run_name':'chat_turn'
        }

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

        status_holder = {'box': None}

        def ai_only_stream() :
            for message_chunk , metadata in chatbot.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ) :

                # model  tool call  -> show status box 
                for tc in getattr(message_chunk, 'tool_call_chunks', None) or [] :
                    if tc.get('name') :
                        label = f"🔧 Using `{tc['name']}` ..."
                        if status_holder['box'] is None :
                            status_holder['box'] = st.status(label, expanded=True)
                        else :
                            status_holder['box'].update(label=label, state='running', expanded=True)

                # tool result -> status complete
                if isinstance(message_chunk, ToolMessage) and status_holder['box'] is not None :
                    status_holder['box'].update(
                        label=f"✅ Used `{message_chunk.name}`", state='complete', expanded=False
                    )

                # streaming AI message only
                if isinstance(message_chunk, AIMessage) and message_chunk.content :
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state['message_history'].append({'role':'assistant','content':ai_message})

    doc_meta = thread_document_metadata(thread_key)                           
    if doc_meta :                                                           
        st.caption(                                                         
            f"Document indexed: {doc_meta.get('filename')} "                  
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"  
        )                                                                   