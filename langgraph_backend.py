from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from typing import TypedDict , Annotated
from langchain_core.messages import HumanMessage , BaseMessage , SystemMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

model = ChatGroq( model="openai/gpt-oss-120b")

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant built by Sumit. "
        "Never claim to be ChatGPT or made by OpenAI. "
        "If asked about your name or model, say you are a chatbot built by Sumit "
        "using LangGraph and Groq."
    )
)

class CHAT_STATE(TypedDict) : 

    messages :  Annotated[list[BaseMessage],add_messages]


def chat_node(state : CHAT_STATE) : 

    # take user query from state
    
    messages = [SYSTEM_PROMPT] + state['messages']

    # send to LLm

    response = model.invoke(messages)

    # response store back to state

    return {'messages' : [response]}


conn = sqlite3.connect(database='chatbot.db',check_same_thread = False)

#checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(CHAT_STATE)

# add node

graph.add_node('chat_node',chat_node)

# add edges 

graph.add_edge(START , 'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads() : 
    all_threads = set()
    for checkpoint in checkpointer.list(None) :
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    
    return list((all_threads))