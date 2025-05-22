import os
from mode import Mode
from console import Console
from argparse import _SubParsersAction
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class DocMode(Mode):

    EXCLUDED_DIRS = {"venv", "__pycache__", ".git", ".idea", ".mypy_cache", ".pytest_cache"}
    INCLUDED_EXTENSIONS = {".py"}

    history: list[BaseMessage] = []

    def __init__(
        self,
        console: Console,
        model: str = "llama3.2:3b",
        system: str = "doc",
        path: str = ".",
        out: str | None = None,
        verbose: bool = False):
        super().__init__(console)

        self.model = model
        self.system = system
        self.path = path
        self.out = out
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        doc_subparser = subparser.add_parser(name)
        doc_subparser.add_argument("--model", type=str, default="llama3.2:3b")
        doc_subparser.add_argument("--system", type=str, default="doc")
        doc_subparser.add_argument("--path", type=str, default=".")
        doc_subparser.add_argument("--out", type=str, default=None)
        doc_subparser.add_argument("--verbose", "-v", action="store_true")

    def run(self):
        # Read system prompt
        system_prompt_path = os.path.join(os.getenv("PROMPTS_DIR"), f"{self.system}.txt")
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

        if self.verbose:
            self.console.system_output(system_prompt)

        # Get the code
        source_files = self._collect_python_files(self.path)
        if not source_files:
            self.console.error("Python files not found.")
            return

        all_code = ""
        for file_path in source_files:
            if self.verbose:
                self.console.info(f" Including : {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
                all_code += f"\n### File: {file_path}\n{code}\n"

        self.history.append(HumanMessage(content=all_code))

        # Load model
        if self.verbose:
            self.console.info(f"Loading model {self.model}...")

        model = init_chat_model(
            self.model,
            model_provider="ollama",
            temperature=1
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | model | StrOutputParser()

        self.console.info("Generating documentation...")

        # Generation
        self.console.bot_start()
        bot_message = ""
        stream = chain.stream({"messages": self.history})
        for chunk in stream:
            bot_message += chunk
            self.console.bot_chunk(chunk)
        self.console.bot_end()

        self.history.append(AIMessage(content=bot_message))

        if self.out:
            with open(self.out, "w", encoding="utf-8") as f:
                f.write(bot_message)
            self.console.info(f"Documentation saved in : {self.out}")

    def _collect_python_files(self, root_dir):
        python_files = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in self.EXCLUDED_DIRS]
            for f in filenames:
                _, ext = os.path.splitext(f)
                if ext in self.INCLUDED_EXTENSIONS:
                    python_files.append(os.path.join(dirpath, f))
        return python_files
