#!/usr/bin/env python3
import signal
from dotenv import load_dotenv
from console import Console
from modes.book_mode import BookMode
from modes.chat_mode import ChatMode
from modes.haiku_mode import HaikuMode
from app import App
from modes.load_book_mode import LoadBookMode
from modes.load_haiku_mode import LoadHaikuMode
from modes.ask_mode import AskMode

load_dotenv()

if __name__ == "__main__":
    # Initialize console
    console = Console()

    # Define signal handler
    def sigkill_handler(sig, frame):
        console.print()
        console.bot_output("Au revoir humain!")
        exit(0)

    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    # Setup app
    app = App(console=console)

    app.use("chat", ChatMode)
    app.use("ask", AskMode)
    app.use("haiku", HaikuMode)
    app.use("load-haiku", LoadHaikuMode)
    app.use("load-book", LoadBookMode)
    app.use("book", BookMode)

    app.run()
