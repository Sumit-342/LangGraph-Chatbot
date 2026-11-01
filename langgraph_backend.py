from __future__ import annotations

from langgraph.graph import StateGraph , START , END
from langchain_groq import ChatGroq
from typing import TypedDict , Annotated , Optional ,Dict , Any
from langchain_core.messages import HumanMessage , BaseMessage , SystemMessage
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import tempfile
import time

from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage ,BaseMessage
from langgraph.prebuilt import ToolNode , tools_condition
import requests
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig 
from langchain_core.embeddings import Embeddings

load_dotenv()

model = ChatGroq( model="openai/gpt-oss-120b")



# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
#=============================================================

class RemoteEmbeddings(Embeddings):

    def __init__(
        self,
        url: str,
        batch_size: int = 16
    ):
        self.url = url
        self.batch_size = batch_size

    def _embed(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        r = requests.post(
            self.url,
            json={"texts": texts},
            timeout=120
        )

        r.raise_for_status()

        return r.json()["embeddings"]

    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        embeddings = []

        for i in range(
            0,
            len(texts),
            self.batch_size
        ):

            batch = texts[
                i:i + self.batch_size
            ]

            embeddings.extend(
                self._embed(batch)
            )

        return embeddings

    def embed_query(
        self,
        text: str
    ) -> list[float]:

        return self._embed(
            [text]
        )[0]


embedding_model = RemoteEmbeddings(os.getenv("EMBEDDING_MODEL_URL"))   

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF
    and store it for the thread.
    """

    if not file_bytes:
        raise ValueError("No PDF bytes received.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Split PDF
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        # Create vector store
        vector_store = FAISS.from_documents(
            chunks,
            embedding_model
        )

        # Retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        # Store per thread
        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename
            or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename
            or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})      



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


@tool
def rag_tool(query: str, config: RunnableConfig):
    """
    Retrieve information from the PDF uploaded
    in the current thread.
    """


    thread_id = config.get("configurable", {}).get("thread_id")

    retriever = _THREAD_RETRIEVERS.get(
        str(thread_id)
    )

    if retriever is None:
        return {
            "error": "No PDF uploaded yet."
        }

    docs = retriever.invoke(query)

    context = [
        doc.page_content
        for doc in docs
    ]

    metadata = [
        doc.metadata
        for doc in docs
    ]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(
            str(thread_id),
            {}
        ).get("filename")
    }


tools = [smart_search, get_stock, rag_tool]

model_with_tool = model.bind_tools(tools=tools)



SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant built by Sumit. "
        "Never claim to be ChatGPT or made by OpenAI. "
        "If asked about your name or model, say you are a chatbot built by Sumit "
        "using LangGraph and Groq. "
        "If the user asks about an uploaded PDF or document, use rag_tool and answer "      
        "from the retrieved context. If it returns an error, tell the user to upload a PDF." 
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