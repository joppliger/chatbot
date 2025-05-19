#!/usr/bin/env python3

import argparse
import signal
from rich.console import Console
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

if __name__ == "__main__":

    SYSTEM_PROMPT_PREFIX = "[bright_black][[/][bold bright_magenta]system[/][bright_black]]:[/]\n"
    HUMAN_PROMPT_PREFIX = "[bright_black][[/][bold bright_green]human[/][bright_black]]:[/]\n"
    BOT_PROMPT_PREFIX = "[bright_black][[/][bold bright_blue]bot[/][bright_black]]:[/]\n"

    console = Console()

    def sigkill_handler(sig, frame):
        console.print(f"\n{BOT_PROMPT_PREFIX}Au revoir humain!")
        exit(0)

    signal.signal(signal.SIGINT, sigkill_handler)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.2:3b")
    parser.add_argument("--system", type=str, default="/usr/var/chatbot/system/default.txt")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Read system prompt
    with open(args.system, "r") as f:
        system_prompt = f.read()

    # Load model
    if args.verbose:
        print(f"Loading model {args.model}...")

    model = init_chat_model(args.model, model_provider="ollama", temperature=1)

    # Load system prompt
    with open(args.system, "r") as f:
        system_prompt = f.read()

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | model | StrOutputParser()

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