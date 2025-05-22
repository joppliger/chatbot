import os
from mode import Mode
from console import Console
from argparse import _SubParsersAction
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class ChatMode(Mode):

    history: list[BaseMessage] = []

    def __init__(
        self, 
        console: Console,
        model: str = "llama3.2:1b",
        system: str = "default", 
        verbose: bool = False):
        super().__init__(console)

        self.model = model
        self.system = system
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        chat_subparser = subparser.add_parser(name)
        chat_subparser.add_argument("--model", type=str, default="llama3.2:1b")
        chat_subparser.add_argument("--system", type=str, default="default")
        chat_subparser.add_argument("--verbose", "-v", action="store_true")

    def run(self):
        # Read system prompt
        system_prompt_path = os.path.join(os.getenv("PROMPTS_DIR"), f"{self.system}.txt")
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()

        # Load model
        if self.verbose:
            self.console.info(f"Loading model {self.model}...")

        model = init_chat_model(
            self.model, 
            model_provider="ollama", 
            temperature=1)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Create chain
        chain = prompt | model | StrOutputParser()

        # Display optional informations
        if self.verbose:
            self.console.system_output(system_prompt)

        # Print system prompt
        while True:
            user_input = self.console.human_input()

            self.history.append(HumanMessage(user_input))

            self.console.bot_start()
            stream = chain.stream({"messages": self.history})
            bot_message = ""
            for chunk in stream:
                bot_message += chunk
                self.console.bot_chunk(chunk)
            self.console.bot_end()

            self.history.append(AIMessage(bot_message))