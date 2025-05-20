import os
from argparse import _SubParsersAction
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from mode import Mode
from console import Console

class HaikuMode(Mode):
    def __init__(
        self, 
        console: Console, 
        verbose: bool = False):

        super().__init__(console)
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        haiku_subparser = subparser.add_parser(name)
        haiku_subparser.add_argument("--verbose", "-v", action="store_true")
        
    def run(self):
        # Load embedding model
        embeddings_model = os.getenv("EMBEDDING_MODEL")

        if self.verbose:
            print(f"Loading embedding model {embeddings_model}...")

        embeddings = OllamaEmbeddings(model=embeddings_model)

        # Create vector store
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=os.getenv("VECTOR_STORE_DATA")
        )

        while True:
            user_input = self.console.human_input()

            self.console.bot_start()
            response = vector_store.similarity_search(query=user_input, k=1)
            if len(response) == 0:
                self.console.bot_chunk("Je ne connais aucun haiku ¯\\_(ツ)_/¯")
                self.console.bot_end()
                continue

            self.console.bot_chunk(response[0].page_content)
            self.console.bot_end()