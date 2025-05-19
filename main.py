#!/usr/bin/env python3
import argparse
import signal
import os
from dotenv import load_dotenv  
from rich.console import Console
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from haikus import haikus

load_dotenv()

if __name__ == "__main__":
    # Define constants
    SYSTEM_PROMPT_PREFIX = "[bright_black][[/][bold bright_magenta]system[/][bright_black]]:[/]\n"
    HUMAN_PROMPT_PREFIX = "[bright_black][[/][bold bright_green]human[/][bright_black]]:[/]\n"
    BOT_PROMPT_PREFIX = "[bright_black][[/][bold bright_blue]bot[/][bright_black]]:[/]\n"

    # Initialize console
    console = Console()

    # Define signal handler
    def sigkill_handler(sig, frame):
        console.print(f"\n{BOT_PROMPT_PREFIX}Au revoir humain!")
        exit(0)

    signal.signal(signal.SIGINT, sigkill_handler)

    # Parse arguments
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="mode", required=True)

    # add chat subparser
    chat_subparser = subparser.add_parser("chat")
    chat_subparser.add_argument("--model", type=str, default="llama3.2:3b")
    chat_subparser.add_argument("--system", type=str, default="default")
    chat_subparser.add_argument("--verbose", "-v", action="store_true")
    
    # add haiku subparser
    haiku_subparser = subparser.add_parser("haiku")
    haiku_subparser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    


    if args.mode == "haiku":

        embeddings_model = "mxbai-embed-large:latest"

        # Load model
        if args.verbose:
            print(f"Loading model {embeddings_model}...")

        embeddings = OllamaEmbeddings(model=embeddings_model)

        # Create vector store
        vector_store = InMemoryVectorStore(embedding=embeddings)

        # Create documents
        documents = [Document(page_content=haiku) for haiku in haikus]

        # Add documents to vector store
        vector_store.add_documents(documents)

        while True:
            user_input = console.input(HUMAN_PROMPT_PREFIX)

            console.print(BOT_PROMPT_PREFIX, end="")
            response = vector_store.similarity_search(query=user_input, k=1)
            console.print(response[0].page_content)
        

    elif args.mode == "chat":

        # Read system prompt
        system_prompt_path = os.path.join(os.getenv("PROMPTS_DIR"), f"system/{args.system}.txt")
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()

        # Load model
        if args.verbose:
            print(f"Loading model {args.model}...")

        model = init_chat_model(args.model, model_provider="ollama", temperature=1)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Create chain
        chain = prompt | model | StrOutputParser()

        # Display optional informations
        if args.verbose:
            console.print(SYSTEM_PROMPT_PREFIX + system_prompt, end="")

        # Print system prompt
        while True:
            user_input = console.input(HUMAN_PROMPT_PREFIX)

            console.print(BOT_PROMPT_PREFIX, end="")
            stream = chain.stream({"messages": [HumanMessage(user_input)]})
            for chunk in stream:
                console.print(chunk, end="")
            print()