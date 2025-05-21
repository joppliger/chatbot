from os import path, getenv

from argparse import _SubParsersAction

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

        for chunk in loader.lazy_load():
            self.console.info(chunk)
            self.console.print('\n\n\n')