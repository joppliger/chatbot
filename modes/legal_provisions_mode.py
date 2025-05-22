from os import getenv

from argparse import _SubParsersAction

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
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

    def run(self):
        system_prompt = """
        Réponds à la question de l’utilisateur en t’appuyant sur le contenu des codes législatifs français 
        (Code civil, Code pénal, Code de l’action sociale et des familles, etc.).

        Voici les extraits pertinents des articles de loi relatifs à la question de l’utilisateur :

        {documents}
        """

        # load VectorStore
        vector_store = Chroma(
            collection_name='legal-provisions',
            embedding_function=OpenAIEmbeddings(
                model = getenv('EMBEDDING_MODEL'),
                api_key = getenv('OPENAI_API_KEY')
            ),
            persist_directory="./store"
        )

        # Load model
        if self.verbose:
            self.console.info(f"Loading model {self.model}...")

        model = init_chat_model(
            self.model, 
            model_provider="openai", 
            temperature=1,
            api_key = getenv('OPENAI_API_KEY')
        )

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Create chain
        chain = prompt | model | StrOutputParser()

        # Display optional informations
        if self.verbose:
            self.console.system_output(system_prompt)

        # Print system prompt
        user_input = self.console.human_input()
        self.history.append(HumanMessage(user_input))

        documents = vector_store.similarity_search(user_input, k=10)
        for document in documents:
            self.console.info(document.page_content)

        self.console.bot_start()
        stream = chain.stream({
            "messages": self.history,
            "documents": documents
        })
        bot_message = ""
        for chunk in stream:
            bot_message += chunk
            self.console.bot_chunk(chunk)
        self.console.bot_end()

        self.history.append(AIMessage(bot_message))