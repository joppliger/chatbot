from typing import Iterator, Tuple

from re import compile, MULTILINE

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class LegalProvisionsLoader():
    RE_LEVELS = {
        "partie": compile(r'^\s*(Partie [^\n]+)', flags=MULTILINE),
        "livre": compile(r'^(Livre [^\n]+)', flags=MULTILINE),
        "titre": compile(r'^(Titre [^\n]+)', flags=MULTILINE),
        "chapitre": compile(r'^(Chapitre [^\n]+)', flags=MULTILINE),
        "article": compile(r'^(Article [^\n]+)', flags=MULTILINE)
    }

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

        self.code, self.partie = self._extract_code_partie()

    def _extract_code_partie(self) -> Tuple[str, str]:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = [next(file) for _ in range(3)]

        match = self.RE_LEVELS["partie"].match(lines[2])
            
        return lines[0].strip(), match.group(1).strip() if match else None

    def lazy_load(self) -> Iterator[Document]:
        current = {
            "livre": None,
            "titre": None,
            "chapitre": None,
            "article": None
        }
        buffer = []

        def flush():
            if current['article'] and buffer:
                yield Document(
                    page_content=''.join(buffer).strip(),
                    metadata={
                        "code": self.code,
                        "partie": self.partie,
                        **current,
                        "source": self.file_path
                    }
                )
        
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for _ in range(3):
                next(file)

            for line in file:
                for level in ("livre", "titre", "chapitre"):
                    match = self.RE_LEVELS[level].match(line)
                    if match:
                        current[level] = match.group(1).strip()
                        break
                else:
                    match = self.RE_LEVELS["article"].match(line)
                    if match:
                        yield from flush()

                        current["article"] = match.group(1).strip()
                        buffer = [line]
                    else:
                        buffer.append(line)
        
        yield from flush()