from os import path, getenv

from argparse import _SubParsersAction

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from mode import Mode
from console import Console
from legal_provisions_loader import LegalProvisionsLoader

class LoadLegalProvisionsMode(Mode):
    def __init__(
        self, 
        console: Console, 
        book: str, 
        verbose: bool = False):
        super().__init__(console)

        self.book = book
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        load_book_subparser = subparser.add_parser(name)
        load_book_subparser.add_argument("book", type=str, help="The book to load")
        load_book_subparser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")

    def run(self):
        self.console.info(f"Loading book {self.book}...")

        loader = LegalProvisionsLoader(self.book)

        # embeddings = OpenAIEmbeddings(
        #     model = getenv('EMBEDDING_MODEL'),
        #     api_key = getenv('OPENAI_API_KEY')
        # )

        # vector_store = Chroma(
        #     collection_name='legal-provisions',
        #     embedding_function=embeddings,
        #     persist_directory=getenv('VECTOR_STORE_DATA')
        # )

        self.console.error(loader.code)
        self.console.error(loader.partie)

        for doc in loader.lazy_load():
            # vector_store.add_documents([doc])
            self.console.print(doc, '\n\n')
            self.console.human_input()

