from argparse import _SubParsersAction
import os
from console import Console
from mode import Mode
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import tool
from langchain.agents import AgentExecutor
from typing import Annotated
from langchain.agents import create_tool_calling_agent
from langgraph.graph import START, END
from langgraph.graph.message import MessageGraph
from langchain_core.language_models.chat_models import BaseChatModel

class GraphMode(Mode):
    def __init__(
        self, 
        console: Console,
        model: str = "llama3.2:3b",
        verbose: bool = False):
        super().__init__(console)

        self.verbose = verbose
        self.model = model

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        agent_subparser = subparser.add_parser(name, help="Run the agent mode")
        agent_subparser.add_argument("--verbose", "-v", action="store_true")
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

    def criticize_factory(self, llm: BaseChatModel):
        def criticize_node(state: list[BaseMessage]) -> BaseMessage:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage("""
                    Tu es un copywriter qui maîtrise les bonnes pratiques de rédaction professionnelles.
                    Tu dois critiquer le dernier message envoyé par l'assistant avant toi à la lumière de tes connaissances en copywriting.
                """),
                MessagesPlaceholder("{messages}")
            ])
            
            chain = prompt | llm

            answer = chain.invoke(state)
            self.console.bot_output(answer.content)
            return answer
        return criticize_node
    
    def should_continue_factory(self, limit: int = 3):
        def should_continue(state: list[BaseMessage]):
            return END if len(state) > limit else "chatbot"
        return should_continue

    def run(self):
        llm = init_chat_model(
            self.model,
            model_provider="openai")

        chatbot_node = self.chatbot_factory(llm)
        criticize_node = self.criticize_factory(llm)
        should_continue = self.should_continue_factory(6)

        graph = MessageGraph()

        graph.add_node("chatbot", chatbot_node)
        graph.add_node("criticize", criticize_node)

        graph.set_entry_point("chatbot")
        graph.add_edge("chatbot", "criticize")
        graph.add_conditional_edges("criticize", should_continue)

        app = graph.compile()

        human_input = self.console.human_input()
        history = app.invoke(human_input)
