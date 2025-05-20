import os
from argparse import _SubParsersAction
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from console import Console
from mode import Mode

class LoadHaikuMode(Mode):
    def __init__(
        self, 
        console: Console, 
        verbose: bool = False,
        file: str = None):

        super().__init__(console)
        self.verbose = verbose
        self.file = file

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        load_haiku_subparser = subparser.add_parser("load-haiku")
        load_haiku_subparser.add_argument("--verbose", "-v", action="store_true")
        load_haiku_subparser.add_argument("--file", type=str, default=None)

    def run(self):
        embeddings_model = os.getenv("EMBEDDING_MODEL")

        if self.verbose:
            print(f"Loading embedding model {embeddings_model}...")

        embeddings = OllamaEmbeddings(model=embeddings_model)

        # Create vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=os.getenv("VECTOR_STORE_DATA")
        )

        if self.file:
            with open(self.file, "r") as f:
                haikus = f.readlines()

            haikus = [haiku.strip() for haiku in haikus]

            for haiku in haikus:
                vector_store.add_texts([haiku])

            self.console.print(f"{len(haikus)} haikus added to vector store.")

        else:
            while True:
                user_input = self.console.human_input()
                vector_store.add_texts([user_input])

                if self.verbose:
                    self.console.bot_start()
                    self.console.bot_chunk("Haiku added to vector store.")
                    self.console.bot_end()
