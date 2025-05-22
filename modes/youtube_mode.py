from argparse import _SubParsersAction
from console import Console
from mode import Mode
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import os
import time
import json
import hashlib
from youtube_transcript_api import YouTubeTranscriptApi
import re
# Imports pour le cache
import langchain
from langchain_community.cache import SQLiteCache

class YoutubeMode(Mode):
    model: str = os.getenv("MODEL_DEFAULT") or "llama3.2:1b"
    system: str = "default"
    history: list[BaseMessage] = []

    def __init__(
        self,
        console: Console,
        url: str,
        transcript: str = None,
        verbose: bool = False,
        model: str = None,
        clear_cache: bool = False):
        super().__init__(console)

        self.url = url
        self.transcript = transcript
        self.verbose = verbose
        if model:
            self.model = model

        # Initialisation du cache LangChain
        cache_path = os.path.join(os.path.dirname(os.getenv("CACHE_DIR")), ".langchain.db")
        if clear_cache and os.path.exists(cache_path):
            if self.verbose:
                self.console.info(f"Suppression du cache LangChain ({cache_path})")
            os.remove(cache_path)

        # Activation du cache LangChain
        langchain.cache = SQLiteCache(database_path=cache_path)
        if self.verbose:
            self.console.info(f"Cache LangChain activé ({cache_path})")

        # Initialisation du cache de résumés
        self.summaries_cache_dir = os.path.join(os.path.dirname(os.getenv("CACHE_DIR")), "summaries_cache")
        os.makedirs(self.summaries_cache_dir, exist_ok=True)

        # Si clear_cache est activé, vider aussi le cache des résumés
        if clear_cache:
            for file in os.listdir(self.summaries_cache_dir):
                if file.endswith('.json'):
                    os.remove(os.path.join(self.summaries_cache_dir, file))
            if self.verbose:
                self.console.info(f"Cache des résumés vidéo vidé")

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        youtube_subparser = subparser.add_parser(name)
        youtube_subparser.add_argument("url", type=str, help="URL de la vidéo Youtube à résumer")
        youtube_subparser.add_argument("--transcript", "-t", type=str, help="Chemin vers un fichier de transcription local (optionnel)")
        youtube_subparser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
        youtube_subparser.add_argument("--model", "-m", type=str, help="Modèle à utiliser (ex: llama3.2:3b)")
        youtube_subparser.add_argument("--clear-cache", "-cc", action="store_true", help="Vider le cache LangChain et des résumés")

    def get_video_id(self, url):
        # Extrait l'ID de la vidéo depuis l'URL
        regex = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
        match = re.search(regex, url)
        return match.group(1) if match else None

    def get_cache_key(self, video_id):
        """Génère une clé de cache basée sur l'ID de la vidéo"""
        return hashlib.md5(video_id.encode()).hexdigest()

    def get_cached_data(self, video_id):
        """Récupère le résumé et la transcription en cache pour une vidéo donnée"""
        cache_key = self.get_cache_key(video_id)
        cache_file = os.path.join(self.summaries_cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if self.verbose:
                    self.console.info(f"Données trouvées en cache pour la vidéo {video_id}")
                return cache_data.get('summary'), cache_data.get('transcript')
            except Exception as e:
                self.console.error(f"Erreur lors du chargement des données en cache : {e}")
                return None, None
        return None, None

    def save_summary_to_cache(self, video_id, summary, transcript=None):
        """Sauvegarde le résumé d'une vidéo dans le cache"""
        cache_key = self.get_cache_key(video_id)
        cache_file = os.path.join(self.summaries_cache_dir, f"{cache_key}.json")

        cache_data = {
            'video_id': video_id,
            'summary': summary,
            'timestamp': time.time()
        }

        # Optionnellement, stocker aussi la transcription
        if transcript:
            cache_data['transcript'] = transcript

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            if self.verbose:
                self.console.info(f"Résumé sauvegardé en cache pour la vidéo {video_id}")
        except Exception as e:
            if self.verbose:
                self.console.error(f"Erreur lors de la sauvegarde du cache : {e}")

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
                    self.console.info("Transcriptions disponibles :")
                    for transcript in transcript_list:
                        self.console.info(f" - {transcript.language_code} | Générée: {transcript.is_generated} | Traduisible: {transcript.is_translatable}")

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
                        self.console.info(f"Impossible de récupérer la transcription française directe: {e}")

                # 2. Sinon, essaye de traduire une transcription en français
                for transcript in transcript_list:
                    if transcript.is_translatable:
                        try:
                            translated = transcript.translate('fr')
                            transcript_data = translated.fetch()
                            texts = [t['text'] if isinstance(t, dict) else getattr(t, 'text', '') for t in transcript_data]
                            texts = [txt.strip() for txt in texts if txt.strip()]
                            if texts:
                                return " ".join(texts)
                        except Exception as e:
                            if self.verbose:
                                self.console.info(f"Erreur lors de la traduction en français: {e}")
                        try:
                            translated = transcript.translate('en')
                            transcript_data = translated.fetch()
                            texts = [t['text'] if isinstance(t, dict) else getattr(t, 'text', '') for t in transcript_data]
                            texts = [txt.strip() for txt in texts if txt.strip()]
                            if texts:
                                return " ".join(texts)
                        except Exception as e:
                            if self.verbose:
                                self.console.info(f"Erreur lors de la traduction en anglais: {e}")

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

        # Récupérer l'ID de la vidéo
        video_id = self.get_video_id(self.url)
        if not video_id:
            self.console.error("URL YouTube invalide.")
            return

        # Vérifier si des données existent déjà en cache
        cached_summary, cached_transcript = self.get_cached_data(video_id)

        # Initialiser transcript à None
        transcript = None

        # Initialiser le modèle ici, avant les branches conditionnelles
        if self.verbose:
            self.console.info(f"Chargement du modèle {self.model}...")

        model = init_chat_model(
            self.model,
            model_provider="ollama",
            temperature=1)

        if cached_summary:
            self.console.info("Utilisation du résumé en cache")
            resume = cached_summary
            self.console.bot_start()
            self.console.info("Résumé de la vidéo :")
            self.console.bot_chunk(resume)
            self.console.bot_end()
            self.console.info("\n---\n")

            # Utiliser la transcription en cache si disponible
            transcript = cached_transcript

            # Si pas de transcription en cache, l'extraire
            if not transcript:
                self.console.info("Transcription non trouvée en cache, extraction en cours...")
                if self.transcript:
                    transcript = self.load_transcript_from_file(self.transcript)
                else:
                    transcript = self.get_transcript(video_id)
        else:
            # Extraire la transcription si aucun résumé n'est en cache
            if self.transcript:
                transcript = self.load_transcript_from_file(self.transcript)
            else:
                if self.verbose:
                    self.console.info(f"Extraction de la transcription de la vidéo...")
                transcript = self.get_transcript(video_id)

            # Vérification finale de la transcription
            if not transcript:
                self.console.error("Impossible de récupérer la transcription.")
                self.console.info("Conseil: Téléchargez manuellement la transcription et utilisez l'option --transcript")
                return

            # Debug de la transcription
            self.console.info(f"Longueur de la transcription : {len(transcript)} caractères")
            self.console.info("Extrait de la transcription :" + transcript[:500])

            # Générer un nouveau résumé
            MAX_TRANSCRIPT_CHARS = 10000
            transcript_for_summary = transcript[:]
            self.console.info(f"Génération d'un nouveau résumé sur {len(transcript_for_summary)} caractères.")

            # Prompt simplifié et explicite pour le résumé
            resume_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Tu es un assistant qui résume des vidéos YouTube à partir de leur transcription."
                ),
                HumanMessage(f"Voici la transcription d'une vidéo YouTube :\n\n{transcript_for_summary}\n\nRésume cette vidéo en français, en 10-15 lignes maximum."),
            ])
            resume_chain = resume_prompt | model | StrOutputParser()
            self.console.bot_start()
            self.console.info("Résumé de la vidéo :")
            resume = ""
            for chunk in resume_chain.stream({}):
                resume += chunk
                self.console.bot_chunk(chunk)
            self.console.bot_end()

            # Sauvegarder le résumé et la transcription en cache
            self.save_summary_to_cache(video_id, resume, transcript)

            self.console.info("\n---\n")

        # 2. Discussion interactive avec l'IA sur la vidéo
        if transcript:
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
            self.console.info("Vous pouvez maintenant discuter avec l'IA à propos de la vidéo.")
            self.console.info(f"La transcription a été divisée en {len(transcript_segments)} segments pour faciliter le traitement.")
            self.console.info("Vous pouvez demander un segment spécifique en tapant 'segment X' (ex: 'segment 2')")

            while True:
                user_input = self.console.human_input()

                # Si l'utilisateur demande un segment spécifique
                segment_match = re.search(r"segment\\s+(\\d+)", user_input.lower())
                if segment_match:
                    segment_num = int(segment_match.group(1)) - 1
                    if 0 <= segment_num < len(transcript_segments):
                        user_input += f"\n\nVoici le segment {segment_num+1} complet:\n{transcript_segments[segment_num]}"
                        self.console.info(f"[Ajout du segment {segment_num+1} à la requête]")

                self.history.append(HumanMessage(user_input))
                stream = chain.stream({"messages": self.history})
                bot_message = ""
                self.console.bot_start()
                for chunk in stream:
                    bot_message += chunk
                    self.console.bot_chunk(chunk)
                self.console.bot_end()
                self.history.append(AIMessage(bot_message))
        else:
            self.console.error("Discussion interactive impossible : Aucune transcription disponible.")
            self.console.info("Conseil: Utilisez l'option --clear-cache pour forcer la récupération de la transcription.")