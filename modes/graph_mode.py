from argparse import _SubParsersAction
import os
from console import Console
from mode import Mode
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph import START, END
from langgraph.graph.message import MessageGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tavily import TavilySearch
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4
import sqlite3

class GraphMode(Mode):
    def __init__(
        self, 
        console: Console,
        model: str = "llama3.2:3b",
        thread: str = None,
        verbose: bool = False):
        super().__init__(console)

        self.verbose = verbose
        self.model = model
        self.thread = thread

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        agent_subparser = subparser.add_parser(name, help="Run the agent mode")
        agent_subparser.add_argument("--verbose", "-v", action="store_true")
        agent_subparser.add_argument("--thread", "-t", default=None)
        agent_subparser.add_argument("--model", type=str, default=os.getenv("DEFAULT_MODEL"))
    
    def chatbot_factory(self, llm: BaseChatModel):
        def chatbot_node(state: list[BaseMessage]) -> BaseMessage:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Tu es un assistant qui répond aux questions de l'utilisateur et qui prend en compte les critiques lorsqu'il y en a."),
                MessagesPlaceholder("{messages}")
            ])
            
            chain = prompt | llm

            answer = chain.invoke(state)
            self.console.bot_output(answer.content)
            return answer
        return chatbot_node

    def criticize_factory(self, llm: BaseChatModel, search_tool: BaseTool):
        def criticize_node(state: list[BaseMessage]) -> BaseMessage:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage("""
                    Tu es un copywriter qui maîtrise les bonnes pratiques de rédaction professionnelles.
                    Tu dois critiquer le dernier message envoyé par l'assistant avant toi à la lumière de tes connaissances en copywriting.
                """),
                MessagesPlaceholder("{messages}")
            ])
            
            chain = prompt | llm.bind_tools(
                tools=[search_tool], 
                tool_choice=search_tool.name)

            return chain.invoke(state)
        return criticize_node
    
    def should_continue_factory(self, limit: int = 3):
        self._have_been_executed = False
        def should_continue(state: list[BaseMessage]):
            if self._have_been_executed:
                return END
            self._have_been_executed = True
            return "chatbot"
            
        return should_continue

    def run(self):
        llm = init_chat_model(
            self.model,
            model_provider="openai")
        
        search_tool = TavilySearch(max_results=3, topic="general")

        if not self.thread:
            memory = InMemorySaver()
        else:
            conn = sqlite3.connect(
                os.path.join(os.getenv("VECTOR_STORE_DATA"), "checkpoint.db"), 
                check_same_thread=False)
            memory = SqliteSaver(conn=conn)

        chatbot_node = self.chatbot_factory(llm)
        criticize_node = self.criticize_factory(llm, search_tool=search_tool)
        search_node = ToolNode(tools=[search_tool])
        should_continue = self.should_continue_factory(6)

        graph = MessageGraph()

        graph.add_node("chatbot", chatbot_node)
        graph.add_node("criticize", criticize_node)
        graph.add_node("search", search_node)

        graph.set_entry_point("chatbot")
        graph.set_finish_point("chatbot")
        #graph.add_edge("chatbot", "criticize")
        #graph.add_edge("criticize", "search")
        #graph.add_conditional_edges("search", should_continue)

        app = graph.compile(checkpointer=memory)

        config = {
            "configurable": {
                "thread_id": self.thread if self.thread else uuid4()
            }
        }

        while True:
            human_input = self.console.human_input()
            history = app.invoke(human_input, config=config)
