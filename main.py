#!/usr/bin/env python3
import signal
from dotenv import load_dotenv
from console import Console
from modes.book_mode import BookMode
from modes.chat_mode import ChatMode
from modes.graph_mode import GraphMode
from modes.haiku_mode import HaikuMode
from app import App
from modes.load_book_mode import LoadBookMode
from modes.load_haiku_mode import LoadHaikuMode
from modes.ask_mode import AskMode
from modes.doc_mode import DocMode
from modes.youtube_mode import YoutubeMode
from modes.agent_mode import AgentMode

load_dotenv()

# Initialize console
console = Console()
# Define signal handler
def sigkill_handler(sig, frame):
    console.bot_output("Au revoir humain!")
    exit(0)

if __name__ == "__main__":
    

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
    app.use("doc", DocMode)
    app.use("youtube", YoutubeMode)
    app.use("agent", AgentMode)
    app.use("graph", GraphMode)

    app.run()
