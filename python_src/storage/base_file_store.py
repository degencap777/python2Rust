from abc import ABCMeta, abstractmethod
from PIL import Image as PILImage
from utils.exceptions import AbstractMethodNotImplementedError

class BaseFileStore(object, metaclass=ABCMeta):
    """Abstract base class to define standard interface for working with local files and cloud storage."""

    @abstractmethod
    def check_if_file_exists(self, path_to_file: str) -> bool:
        raise AbstractMethodNotImplementedError()
    
    @abstractmethod
    def check_if_dir_exists(self, path_to_directory: str) -> bool:
        raise AbstractMethodNotImplementedError()
    
    @abstractmethod
    def create_dir(self, path_to_directory: str, exist_ok: bool = True) -> None:
        raise AbstractMethodNotImplementedError()
    
    @abstractmethod
    def download_dir_files_to_local(self, path_to_directory: str, local_path: str) -> None:
        raise AbstractMethodNotImplementedError()

    @abstractmethod
    def download_file_to_local(self, path_to_file: str, local_path: str) -> None:
        raise AbstractMethodNotImplementedError()
    
    @abstractmethod
    def read_text_file(self, path_to_file: str) -> str:
        raise AbstractMethodNotImplementedError()

    @abstractmethod
    def read_jsonl_file(self, path_to_file: str) -> list[dict]:
        raise AbstractMethodNotImplementedError()

    @abstractmethod
    def read_json_file(self, path_to_file: str) -> dict:
        raise AbstractMethodNotImplementedError()

    @abstractmethod
    def read_image(self, file_name: str) -> PILImage.Image:
        raise AbstractMethodNotImplementedError()

    @abstractmethod
    def read_text_files_from_dir(self, path_to_directory: str) -> list[str]:
        raise AbstractMethodNotImplementedError()

    @abstractmethod
    def read_images_from_dir(self, path_to_directory: str) -> list[PILImage.Image]:
        raise AbstractMethodNotImplementedError()
    
    @abstractmethod
    def read_bytes(self, file_name: str) -> bytes:
        raise AbstractMethodNotImplementedError()
    
    @abstractmethod
    def save_file(self, src_file: str, dest_file: str, overwrite: bool) -> None:
        raise AbstractMethodNotImplementedError()

    @abstractmethod
    def write_bytes(self, file_name: str, data: bytes) -> None:
        raise AbstractMethodNotImplementedError()
    
    @abstractmethod
    def final_file_uri(self, file_path: str) -> None:
        raise AbstractMethodNotImplementedError()

