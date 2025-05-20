from argparse import _SubParsersAction
from console import Console
from mode import Mode

class LoadBookMode(Mode):
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
