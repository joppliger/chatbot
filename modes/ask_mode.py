import os
from mode import Mode
from console import Console
from argparse import _SubParsersAction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

class AskMode(Mode):
    def __init__(
        self, 
        console: Console,
        model: str = "llama3.2:1b",
        system: str = "default", 
        out: str|None = None,
        data: list[str]|None = None,
        verbose: bool = False):
        super().__init__(console)

        self.model = model
        self.system = system
        self.out = out
        self.data = data
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        chat_subparser = subparser.add_parser(name)
        chat_subparser.add_argument("--model", type=str, default="llama3.2:1b")
        chat_subparser.add_argument("--system", type=str, default="default")
        chat_subparser.add_argument("--verbose", "-v", action="store_true")
        chat_subparser.add_argument("--out", type=str, default=None)
        chat_subparser.add_argument("--data", "-d", action="append", type=str, default=None)

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

        # Parse data
        system_data = {}
        if self.data:
            for data in self.data:
                key, value = data.split("=")
                system_data[key] = value
                if self.verbose:
                    self.console.info(f"Data: {key}: {value}")

        # Display optional informations
        if self.verbose:
            self.console.system_output(system_prompt)

        system_prompt = SystemMessagePromptTemplate.from_template(system_prompt)

        try:
            system_prompt_with_data = system_prompt.format(**system_data)
        except Exception as e:
            self.console.error(f"System prompt require additional data : {e}")
            return

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            system_prompt_with_data,
            HumanMessagePromptTemplate.from_template("{request}"),
        ])

        # Create chain
        chain = prompt | model | StrOutputParser()

        # Print system prompt
        user_input = self.console.human_input()

        stream = chain.stream({"request": user_input })

        self.console.bot_start()

        if self.out:
            with open(self.out, "w") as f:
                for chunk in stream:
                    f.write(chunk)
                    self.console.bot_chunk(chunk)
                f.write("\n")
        else:
            for chunk in stream:
                self.console.bot_chunk(chunk)

        self.console.bot_end()

        if self.verbose and self.out:
            self.console.info(f"Output saved to {self.out}")

        