from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from typing import TypedDict , Annotated
from langchain_core.messages import HumanMessage , BaseMessage , SystemMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage ,BaseMessage
from langgraph.prebuilt import ToolNode , tools_condition
import requests
import os

load_dotenv()

model = ChatGroq( model="openai/gpt-oss-120b")
#=============================================================
# Tools

tavily_search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))

duck_duck_go_search = DuckDuckGoSearchRun(region="us-en")


@tool
def smart_search(query : str) -> dict :
    """
    Tavliy as Primary Search Tool , DuckDuckGo as Fallback
    """

    try :
        result = tavily_search.run(query)
        return{"source": "Tavily","result" : result}

    except Exception as e :
        result = duck_duck_go_search.run(query)
        return{"source":"DuckDuckGo","result":result}


@tool
def get_stock(symbol: str) -> dict:
    """
    Fetch latest stock price of a given symbol (e.g. AAPL, TSLA)
    using Alpha Vantage free API.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()

tools = [smart_search, get_stock]

model_with_tool = model.bind_tools(tools=tools)



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

    response = model_with_tool.invoke(messages)

    # response store back to state

    return {'messages' : [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database='chatbot.db',check_same_thread = False)

#checkpointer
checkpointer = SqliteSaver(conn=conn)


# graph

graph = StateGraph(CHAT_STATE)

# add node

graph.add_node('chat_node',chat_node)
graph.add_node("tools",tool_node)

# add edges 

graph.add_edge(START , 'chat_node')
graph.add_conditional_edges('chat_node',tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads() : 
    all_threads = set()
    for checkpoint in checkpointer.list(None) :
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    
    return list((all_threads))