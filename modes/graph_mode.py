from argparse import _SubParsersAction
import os
from console import Console
from mode import Mode
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import AIMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END
from langgraph.graph.message import MessageGraph
from langgraph.graph import StateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_tavily import TavilySearch
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4
from typing import TypedDict
from pydantic import BaseModel, Field
import sqlite3

class MyState(TypedDict):
    user_request: str = None
    tweet: str = None
    rating: int = None
    critic: str = None

class Critic(BaseModel):
    rating: int = Field(description="Une note de 0 à 100 représentant la qualité du tweet")
    critic: str = Field(description="Une critique du tweet qui contient ses points forts et ses points faibles")

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
        def chatbot_node(state: MyState) -> MyState:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage("""
                Tu es un générateur de tweet de qualité. Tu écris des tweet avec le niveau de qualité d'usage sur Twitter.
                """),
                HumanMessagePromptTemplate.from_template("{user_request}")
            ])
            
            chain = prompt | llm

            answer = chain.invoke(state)

            self.console.bot_output(answer.content)
            state["tweet"] = answer.content
            return state
        return chatbot_node

    def loan_factory(self, llm: BaseChatModel):
        def loan_node(state: MyState):
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage("""
                    Tu es un critique de tweet. 
                    Ton objectif est d'avoir un tweet viral. 
                    Pour cela tu dois critiquer le tweet en indiquant ses points forts et ses points faibles.
                              
                    Quels sont les critères d'un bon tweet :
                    - Un bon tweet a des fautes d'orthographe
                    - Il doit faire réagir avec un avis tranché et caricatural
                    - Il est pas trop long
                    - Il contient des hashtags
                    - Il est limité a 256 caractères
                    - Il pratique l'ironie et l'humour noir
                    - Il n'invente aucune blague, il réutilise des blagues bien usées
                              
                    # Strucure de la réponse :
                    rating: contient la note de 0 à 100 sur la qualité du tweet. 0 c'est la plus mauvaise qualité, 100 c'est parfait.
                    critic: Un commentaire détaillé, composé de recommandations actionnables pour améliorer le tweet. Mentionne les points forts et les points faibles.
                """),
                HumanMessagePromptTemplate.from_template("{tweet}")
            ])
            chain = prompt |llm.with_structured_output(Critic)

            response = chain.invoke({ "tweet": state["tweet"] })

            state["rating"] = response.rating
            state["critic"] = response.critic

            return state
        return loan_node 

    def should_send_factory(self, treshold: int):
        def should_send(state: MyState):
            if state["rating"] >= treshold:
                return END 
            return "chatbot"
        return should_send

    def run(self):
        llm = init_chat_model(
            self.model,
            model_provider="openai")

        if not self.thread:
            memory = InMemorySaver()
        else:
            conn = sqlite3.connect(
                os.path.join(os.getenv("VECTOR_STORE_DATA"), "checkpoint.db"), 
                check_same_thread=False)
            memory = SqliteSaver(conn=conn)

        chatbot_node = self.chatbot_factory(llm)
        loan_node = self.loan_factory(llm)
        should_send = self.should_send_factory(70)

        graph = StateGraph(MyState)

        graph.add_node("chatbot", chatbot_node)
        graph.add_node("loan", loan_node)
        

        graph.set_entry_point("chatbot")
        graph.add_edge("chatbot", "loan")
        graph.add_conditional_edges("loan", should_send)
        graph.set_finish_point("chatbot")

        app = graph.compile(checkpointer=memory)

        config = {
            "configurable": {
                "thread_id": self.thread if self.thread else uuid4()
            }
        }

        while True:
            human_input = self.console.human_input()
            initial_state = MyState(user_request=human_input)
            history = app.invoke(initial_state, config=config)
