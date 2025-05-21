from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
import re

class YoutubeMode(Mode):
    model: str = os.getenv("EMBEDDING_MODEL") or "llama3.2:1b"
    system: str = "default"
    history: list[BaseMessage] = []

    def __init__(
        self,
        console: Console,
        url: str,
        transcript: str = None,  # Renommé pour correspondre à argparse
        verbose: bool = False):
        super().__init__(console)

        self.url = url
        self.transcript = transcript  # Nom cohérent
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        youtube_subparser = subparser.add_parser(name)
        youtube_subparser.add_argument("url", type=str, help="URL de la vidéo Youtube à résumer")
        youtube_subparser.add_argument("--transcript", "-t", type=str, help="Chemin vers un fichier de transcription local (optionnel)")
        youtube_subparser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")

    def get_video_id(self, url):
        # Extrait l'ID de la vidéo depuis l'URL
        regex = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
        match = re.search(regex, url)
        return match.group(1) if match else None

    def load_transcript_from_file(self, file_path):
        """Charge une transcription depuis un fichier texte local"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            if self.verbose:
                self.console.info(f"Transcription chargée depuis {file_path}")
            return transcript
        except Exception as e:
            self.console.error(f"Erreur lors du chargement du fichier de transcription : {e}")
            return None

    def get_transcript(self, video_id, retries=3, delay=2):
        """Récupère la transcription avec gestion des retries en cas d'erreur de connexion"""
        for attempt in range(retries):
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                if self.verbose:
                    print("Transcriptions disponibles :")
                    for transcript in transcript_list:
                        print(f" - {transcript.language_code} | Générée: {transcript.is_generated} | Traduisible: {transcript.is_translatable}")

                # 1. Essaye de récupérer directement la transcription en français
                try:
                    transcript = transcript_list.find_transcript(['fr'])
                    transcript_data = transcript.fetch()
                    texts = [t['text'] if isinstance(t, dict) else getattr(t, 'text', '') for t in transcript_data]
                    texts = [txt.strip() for txt in texts if txt.strip()]  # Filtre les textes vides
                    if texts:
                        return " ".join(texts)
                except Exception as e:
                    if self.verbose:
                        print(f"Impossible de récupérer la transcription française directe: {e}")

                # 2. Sinon, essaye de traduire une transcription en français
                for transcript in transcript_list:
                    if transcript.is_translatable:
                        try:
                            translated = transcript.translate('fr')
                            transcript_data = translated.fetch()
                            texts = [t['text'] if isinstance(t, dict) else getattr(t, 'text', '') for t in transcript_data]
                            texts = [txt.strip() for txt in texts if txt.strip()]  # Filtre les textes vides
                            if texts:
                                return " ".join(texts)
                        except Exception as e:
                            if self.verbose:
                                print(f"Erreur lors de la traduction en français: {e}")
                        try:
                            translated = transcript.translate('en')
                            transcript_data = translated.fetch()
                            texts = [t['text'] if isinstance(t, dict) else getattr(t, 'text', '') for t in transcript_data]
                            texts = [txt.strip() for txt in texts if txt.strip()]  # Filtre les textes vides
                            if texts:
                                return " ".join(texts)
                        except Exception as e:
                            if self.verbose:
                                print(f"Erreur lors de la traduction en anglais: {e}")

                self.console.error("Aucune transcription exploitable trouvée pour cette vidéo.")
                return None

            except Exception as e:
                if attempt < retries - 1:
                    self.console.info(f"Tentative {attempt+1}/{retries} échouée : {e}")
                    self.console.info(f"Nouvelle tentative dans {delay} secondes...")
                    time.sleep(delay)
                    # Augmente le délai pour les tentatives suivantes
                    delay *= 2
                else:
                    self.console.error(f"Erreur lors de la récupération de la transcription : {e}")
                    return None

    def run(self):
        # Charge le prompt système
        system_prompt_path = os.path.join(os.getenv("PROMPTS_DIR"), f"system/{self.system}.txt")
        with open(system_prompt_path, "r") as f:
            system_prompt = f.read()

        # Récupération de la transcription
        transcript = None

        # Si un fichier de transcription est fourni, on l'utilise en priorité
        if self.transcript:  # Utilise self.transcript au lieu de self.transcript_file
            transcript = self.load_transcript_from_file(self.transcript)
        # Sinon, on essaie de récupérer la transcription depuis YouTube
        else:
            if self.verbose:
                self.console.info(f"Extraction de la transcription de la vidéo...")

            video_id = self.get_video_id(self.url)
            if not video_id:
                self.console.error("URL YouTube invalide.")
                return

            transcript = self.get_transcript(video_id)

        # Vérification finale de la transcription
        if not transcript:
            self.console.error("Impossible de récupérer la transcription.")
            self.console.info("Conseil: Téléchargez manuellement la transcription et utilisez l'option --transcript")
            return

        # Debug de la transcription
        print(f"Longueur de la transcription : {len(transcript)} caractères")
        print("Extrait de la transcription :", transcript[:500])

        if self.verbose:
            self.console.info(f"Transcription récupérée, chargement du modèle {self.model}...")

        model = init_chat_model(
            self.model,
            model_provider="ollama",
            temperature=1)

        # 1. Résumé automatique de la vidéo
        # Tronquer la transcription pour le résumé (limite de contexte)
        MAX_TRANSCRIPT_CHARS = 10000
        transcript_for_summary = transcript[:]
        print(f"Résumé sur {len(transcript_for_summary)} caractères.")

        # Prompt simplifié et explicite pour le résumé
        resume_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Tu es un assistant qui résume des vidéos YouTube à partir de leur transcription."
            ),
            HumanMessage(f"Voici la transcription d'une vidéo YouTube :\n\n{transcript_for_summary}\n\nRésume cette vidéo en français, en 10-15 lignes maximum."),
        ])
        resume_chain = resume_prompt | model | StrOutputParser()
        self.console.bot_start()
        print("Résumé de la vidéo :")
        resume = ""
        for chunk in resume_chain.stream({}):
            resume += chunk
            self.console.bot_chunk(chunk)
        self.console.bot_end()
        print("\n---\n")

        # 2. Discussion interactive avec l'IA sur la vidéo
        # Pour la discussion, on peut utiliser la transcription complète
        # mais on divise en segments pour éviter de dépasser le contexte
        SEGMENT_SIZE = 8000  # Taille approximative d'un segment en caractères
        transcript_segments = [transcript[i:i+SEGMENT_SIZE] for i in range(0, len(transcript), SEGMENT_SIZE)]

        # Prompt pour la discussion, avec contexte du système original
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                system_prompt + "\n\nVoici la transcription de la vidéo YouTube (en segments) :\n" +
                "\n".join([f"[Segment {i+1}] {segment[:500]}..." for i, segment in enumerate(transcript_segments)])
            ),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | model | StrOutputParser()

        if self.verbose:
            self.console.system_output(self.system)

        self.console.bot_start()
        print("Vous pouvez maintenant discuter avec l'IA à propos de la vidéo. Tapez 'exit' pour quitter.")
        print(f"La transcription a été divisée en {len(transcript_segments)} segments pour faciliter le traitement.")
        print("Vous pouvez demander un segment spécifique en tapant 'segment X' (ex: 'segment 2')")

        while True:
            user_input = input("Vous: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                break

            # Si l'utilisateur demande un segment spécifique
            segment_match = re.search(r"segment\s+(\d+)", user_input.lower())
            if segment_match:
                segment_num = int(segment_match.group(1)) - 1
                if 0 <= segment_num < len(transcript_segments):
                    user_input += f"\n\nVoici le segment {segment_num+1} complet:\n{transcript_segments[segment_num]}"
                    print(f"[Ajout du segment {segment_num+1} à la requête]")

            self.history.append(HumanMessage(user_input))
            stream = chain.stream({"messages": self.history})
            bot_message = ""
            for chunk in stream:
                bot_message += chunk
                self.console.bot_chunk(chunk)
            self.console.bot_end()
            self.history.append(AIMessage(bot_message))