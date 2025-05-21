import os
from mode import Mode
from console import Console
from argparse import _SubParsersAction
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class BookMode(Mode):

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
        system_prompt = """
        Répond à la question de l'utilisateur en te basant sur
        le contenu du livre « Coder Proprement » écris par Robert C. Martin.

        Voici quelques extraits du livre relatifs à la question 
        de l'utilisateur :

        {documents}
        """

        # load VectorStore
        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
            persist_directory="./.store"
        )

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
        user_input = self.console.human_input()
        self.history.append(HumanMessage(user_input))

        documents = vector_store.similarity_search(user_input, k=4)
        for document in documents:
            self.console.info(document.page_content)

        self.console.bot_start()
        stream = chain.stream({
            "messages": self.history,
            "documents": documents
        })
        bot_message = ""
        for chunk in stream:
            bot_message += chunk
            self.console.bot_chunk(chunk)
        self.console.bot_end()

        self.history.append(AIMessage(bot_message))