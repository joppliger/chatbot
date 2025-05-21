import os
from mode import Mode
from console import Console
from argparse import _SubParsersAction
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

class CVMode(Mode):

    history: list[BaseMessage] = []

    def __init__(
        self, 
        console: Console,
        model: str = "llama3.2:1b",
        system: str = "default", 
        verbose: bool = False):
        super().__init__(console)

        self.model = model
        self.system = system
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        chat_subparser = subparser.add_parser(name)
        chat_subparser.add_argument("--model", type=str, default="llama3.2:1b")
        chat_subparser.add_argument("--system", type=str, default="default")
        chat_subparser.add_argument("--verbose", "-v", action="store_true")

    def run(self):
        # System prompt
        system_prompt = """
            Tu es un recruteur expérimenté. Sur la base du CV fourni ci-dessous, rédige une série de questions d’entretien pertinentes pour évaluer le candidat.
            Les questions doivent couvrir les aspects suivants :
            - Questions techniques liées au domaine du candidat
            - Questions comportementales (soft skills, travail en équipe, etc.)
            - Questions de motivation ou de culture d’entreprise
            Voici le CV de l'utilisateur :
            {documents}
        """

        # Load VectorStore
        vector_store = Chroma(
            embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest"),
            persist_directory="./.store"
        )

        # Requête pour orienter la recherche des documents pertinents
        query = "Génère des questions d’entretien à partir du CV"
        documents = vector_store.similarity_search(query, k=4)

        # Affichage des documents récupérés (optionnel)
        if self.verbose:
            self.console.info("Documents chargés :")
            for document in documents:
                self.console.info(document.page_content)

        # Chargement du modèle
        if self.verbose:
            self.console.info(f"Chargement du modèle {self.model}...")

        model = init_chat_model(
            self.model,
            model_provider="ollama",
            temperature=0.7,
        )

        # Construction du prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Construction de la chaîne
        chain = prompt | model | StrOutputParser()

        # Message utilisateur simulé pour lancer la génération
        user_message = HumanMessage(content="Génère des questions d’entretien adaptées à mon CV.")
        self.history.append(user_message)

        # Exécution de la chaîne
        self.console.bot_start()
        stream = chain.stream({
            "messages": self.history,
            "documents": documents
        })

        # Affichage du flux de réponse
        bot_message = ""
        for chunk in stream:
            bot_message += chunk
            self.console.bot_chunk(chunk)
        self.console.bot_end()

        self.history.append(AIMessage(bot_message))
