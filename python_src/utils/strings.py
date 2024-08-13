from pathlib import Path
from urllib.parse import urlparse
from unidecode import unidecode

class StringUtils:

    @staticmethod
    def normalize_special_characters(text: str) -> str:
            if not text:
                return text
            text = StringUtils().fix_line_breaks(text)
            return StringUtils().fix_apostrophes(text)
    
    @staticmethod
    def fix_line_breaks(text: str) -> str:
            text = '\n'.join(text.splitlines())
            text = text.replace('\r', '')
            return text

    @staticmethod
    def fix_apostrophes(text: str) -> str:
            if not text:
                return text
            text = unidecode(text)
            return text.replace('\\u2019', "'")
    
    @staticmethod
    def parse_filename_with_extension_from_uri(uri: str) -> str:
        """Parses the file name from a URI."""
        parts = urlparse(uri)
        path = Path(parts.path)
        return path.stem + path.suffix
         