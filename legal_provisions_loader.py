from typing import Iterator, Tuple

from re import compile, MULTILINE

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class LegalProvisionsLoader():
    RE_LEVELS = {
        "partie": compile(r'^\s*(Partie [^\n]+)', flags=MULTILINE),
        "livre": compile(r'^\s*(Livre [^\n]+)', flags=MULTILINE),
        "titre": compile(r'^(Titre [^\n]+)', flags=MULTILINE),
        "chapitre": compile(r'^(Chapitre [^\n]+)', flags=MULTILINE),
        "article": compile(r'^(Article [^\n]+)', flags=MULTILINE)
    }

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

        self.partie, self.livre = self._extract_partie_livre()

    def _extract_partie_livre(self) -> Tuple[str, str]:
        partie = livre = None

        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if not partie:
                    match = self.RE_LEVELS["partie"].match(line)
                    if match:
                        partie = match.group(1).strip()
                    
                if not livre:
                    match = self.RE_LEVELS["livre"].match(line)
                    if match:
                        livre = match.group(1).strip()

                if partie and livre:
                    break
            
        return partie, livre

    def lazy_load(self) -> Iterator[Document]:
        hierarchy = {
            "titre": None,
            "chapitre": None,
            "article": None
        }
        buffer = []

        def flush():
            if hierarchy['article'] and buffer:
                yield Document(
                    page_content=''.join(buffer).strip(),
                    metadata={
                        "partie": self.partie,
                        "livre": self.livre,
                        **hierarchy,
                        "source": self.file_path
                    }
                )
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                for level in ("titre", "chapitre"):
                    match = self.RE_LEVELS[level].match(line)
                    if match:
                        hierarchy[level] = match.group(1).strip()
                        break
                else:
                    match = self.RE_LEVELS["article"].match(line)
                    if match:
                        yield from flush()

                        hierarchy["article"] = match.group(1).strip()
                        buffer.clear()
                        buffer.append(line)
                    else:
                        buffer.append(line)
        
        yield from flush()