from argparse import _SubParsersAction
import os
from console import Console
from mode import Mode
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage
from langchain.agents import tool
from langchain.agents import AgentExecutor
from typing import Annotated
from langchain.agents import create_tool_calling_agent

class AgentMode(Mode):
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

    def run(self):
        
        @tool
        def multiplication(
            a: Annotated[int, "un entier à multiplier par b"], 
            b: Annotated[int, "un entier avec lequel multiplier a"]) -> int:
            """
            Cette fonction retourne un entier résultat du produit de l'entier a par l'entier b
            """
            return a * b

        llm = init_chat_model(
            self.model,
            model_provider="ollama")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])

        agent = create_tool_calling_agent(llm, [multiplication], prompt=prompt)

        agent_executor = AgentExecutor(agent=agent, tools=[multiplication])
        
        human_input = self.console.human_input()

        response = agent_executor.invoke({"input": human_input})

        print(response)