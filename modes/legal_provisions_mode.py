from os import getenv

from argparse import _SubParsersAction

from langchain.chat_models import init_chat_model
from langchain.output_parsers import BooleanOutputParser

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma

from mode import Mode
from console import Console

class LegalProvisionsMode(Mode):
    history: list[BaseMessage] = []

    def __init__(
        self, 
        console: Console,
        model: str = "gpt-4o-mini",
        system: str = "default", 
        verbose: bool = False
    ):
        super().__init__(console)

        self.model = model
        self.system = system
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        chat_subparser = subparser.add_parser(name)
        chat_subparser.add_argument("--model", type=str, default="gpt-4o-mini")
        chat_subparser.add_argument("--system", type=str, default="default")
        chat_subparser.add_argument("--verbose", "-v", action="store_true")

    def __should_retrieve_from_history_only(
        self,
        model
    ):
        message_prompt = """
        Compte tenu de l'historique de conversation, réponds par 'oui' ou par 'non' s'il est possible de répondre à la question de l'utilisateur en fonction des informations déjà disponible dans l'historique.\n
        Réponds excclusivement soit 'oui', soit 'non' et rien de plus.
        """

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(message_prompt),
            MessagesPlaceholder(variable_name="history"),
        ])

        chain = prompt | model | BooleanOutputParser(false_val='non', true_val='oui')
        response = chain.invoke({ 'history': self.history })

        return response


    def run(self):
        # Load model
        if self.verbose:
            self.console.info(f"Loading model {self.model}...")

        model = init_chat_model(
            self.model,
            model_provider="openai",
            api_key = getenv('OPENAI_API_KEY')
        )

        # Load vector store
        if self.verbose:
            self.console.info(f"Loading embedding {getenv('EMBEDDING_MODEL')}")
        
        embeddings = OpenAIEmbeddings(
            model=getenv('EMBEDDING_MODEL'),
            api_key=getenv('OPENAI_API_KEY')
        )

        vector_store = Chroma(
            collection_name='legal-provisions',
            embedding_function=embeddings,
            persist_directory="./.store"
        )

        # System prompt
        system_prompt = """
        Réponds à la question de l’utilisateur en t’appuyant sur le contenue de l'historique de conversation et en t'appuyant sur les extraits du code de l'action sociale et des familles fourni ci-dessous.

        Voici les extraits pertinents des articles de loi relatifs à la question de l’utilisateur :

        {documents}
        """

        if self.verbose:
            self.console.system_output(system_prompt)

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Create chain
        chain = prompt | model | StrOutputParser()

        while True:
            user_input = self.console.human_input()
            self.history.append(HumanMessage(user_input))
            
            should_retrieve = self.__should_retrieve_from_history_only(model=model)
            if self.verbose:
                self.console.info(f"Should retrive from history only : {should_retrieve}")

            if not should_retrieve:
                documents = vector_store.similarity_search(query=user_input, k=5)

                if self.verbose:
                    for document in documents:
                        self.console.info(f'{document} \n')
                
                self.console.bot_start()
                stream = chain.stream({
                    "messages": self.history,
                    "documents": documents
                })
                bot_message = ""
                for chunk in stream:
                    bot_message += chunk
                    self.console.bot_chunk(chunk=chunk)
                self.console.bot_end()

                self.history.append(AIMessage(content=bot_message))
            else:
                self.console.bot_start()
                stream = chain.stream({
                    "messages": self.history,
                    "documents": None
                })
                bot_message = ""
                for chunk in stream:
                    bot_message += chunk
                    self.console.bot_chunk(chunk=chunk)
                self.console.bot_end()

                self.history.append(AIMessage(content=bot_message))