from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
import os
from youtube_transcript_api import YouTubeTranscriptApi
import re
from langchain_core.messages import BaseMessage



class YoutubeMode(Mode):
    model: str = os.getenv("EMBEDDING_MODEL") or "llama3.2:3b"
    system: str = "default"
    history: list[BaseMessage] = []
    
    def __init__(
        self, 
        console: Console, 
        url: str, 
        verbose: bool = False):
        super().__init__(console)

        self.url = url
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        youtube_subparser = subparser.add_parser(name)
        youtube_subparser.add_argument("url", type=str, help="Resume une vidéo Youtube")

    def get_video_id(self, url):
        # Extrait l'ID de la vidéo depuis l'URL
        regex = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
        match = re.search(regex, url)
        return match.group(1) if match else None

    def get_transcript(self, video_id):
        # Récupère la transcription de la vidéo
        try:
            # Essaye d'abord en français, puis en anglais, puis sans langue
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['fr'])
            except Exception:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                except Exception:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([t['text'] for t in transcript])
        except Exception as e:
            self.console.error(f"Erreur lors de la récupération de la transcription : {e}")
            return None

    def run(self):
        system_prompt_path = os.path.join(os.getenv("PROMPTS_DIR"), f"system/{self.system}.txt")
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()

        if self.verbose:
            self.console.info(f"Extraction de la transcription de la vidéo...")

        video_id = self.get_video_id(self.url)
        if not video_id:
            self.console.error("URL YouTube invalide.")
            return

        transcript = self.get_transcript(video_id)
        if not transcript:
            self.console.error("Impossible de récupérer la transcription.")
            return

        if self.verbose:
            self.console.info(f"Transcription récupérée, chargement du modèle {self.model}...")

        model = init_chat_model(
            self.model, 
            model_provider="ollama", 
            temperature=1)

        # Crée un prompt pour résumer la vidéo
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessage(f"Voici la transcription d'une vidéo YouTube :\n{transcript}\n\nPeux-tu en faire un résumé ?"),
            MessagesPlaceholder(variable_name="messages"),
        ])

        print(prompt)
        chain = prompt | model | StrOutputParser()

        if self.verbose:
            self.console.system_output(self.system)

        self.console.bot_start()
        stream = chain.stream({"messages": self.history})
        bot_message = ""
        for chunk in stream:
            bot_message += chunk
            self.console.bot_chunk(chunk)
        self.console.bot_end()

        self.history.append(AIMessage(bot_message))