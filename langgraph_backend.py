from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from typing import TypedDict , Annotated
from langchain_core.messages import HumanMessage , BaseMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = ChatGroq( model="openai/gpt-oss-120b")

class CHAT_STATE(TypedDict) : 

    messages :  Annotated[list[BaseMessage],add_messages]


def chat_node(state : CHAT_STATE) : 

    # take user query from state
    
    messages = state['messages']

    # send to LLm

    response = model.invoke(messages)

    # response store back to state

    return {'messages' : [response]}


check_pointer = InMemorySaver()

graph = StateGraph(CHAT_STATE)

# add node

graph.add_node('chat_node',chat_node)

# add edges 

graph.add_edge(START , 'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=check_pointer)
