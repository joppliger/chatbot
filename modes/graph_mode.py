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
                SystemMessage(content="You are a helpful assistant."),
                HumanMessagePromptTemplate.from_template("{input}"),
            ])
            
            chain = prompt | llm
            return chain.invoke({"input": state[0].content})
        return chatbot_node


    def run(self):

        llm = init_chat_model(
            self.model,
            model_provider="openai")

        chatbot_node = self.chatbot_factory(llm)

        graph = MessageGraph()

        graph.add_node("chatbot", chatbot_node)

        graph.set_entry_point("chatbot")
        graph.set_finish_point("chatbot")

        app = graph.compile()

        human_input = self.console.human_input()
        history = app.invoke(human_input)

        last_message = history[-1]
        self.console.bot_output(last_message.content)