from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


class LoadCVMode(Mode):
    def __init__(
        self, 
        console: Console, 
        cv: str, 
        verbose: bool = False):
        super().__init__(console)

        self.cv = cv
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        load_cv_subparser = subparser.add_parser(name)
        load_cv_subparser.add_argument("cv", type=str, help="The cv to load")
        load_cv_subparser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")

    def run(self):
        self.console.info(f"Loading cv {self.cv}...")

        # Create vector store
        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
            persist_directory="./.store"
        )

        # Loading
        text_splitter = SemanticChunker(
            embeddings=OllamaEmbeddings(model="mxbai-embed-large:latest"),
        )
        
        loader = PyPDFLoader(self.cv)
        for page in loader.lazy_load():
            chunks = text_splitter.split_documents([page])
            vector_store.add_documents(chunks)
 
 