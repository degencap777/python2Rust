import json
import os
from PIL import Image as PILImage
from storage.base_file_store import BaseFileStore
from utils.exceptions import InvalidArgsError
from utils.strings import StringUtils

class LocalStorage(BaseFileStore):
    """Basic file system I/O for local drives."""

    def __repr__(self) -> str:
        return f'Local file storage'

    def check_if_file_exists(self, path_to_file: str) -> bool:
        return os.path.isfile(path_to_file)

    def check_if_dir_exists(self, path_to_directory: str) -> bool:
        return os.path.isdir(path_to_directory)

    def create_dir(self, path_to_directory: str, exist_ok: bool = True) -> None:
        os.makedirs(path_to_directory, exist_ok=exist_ok)

    def download_file_to_local(self, path_to_file: str, local_path: str) -> None:
        bytes = self.read_bytes(path_to_file)
        self.write_bytes(local_path, bytes)

    def download_dir_files_to_local(self, path_to_directory: str, local_path: str) -> None:
        if not os.path.isdir(path_to_directory):
            raise ValueError(f"{path_to_directory} is not a directory")
        os.makedirs(local_path, exist_ok=True)
        for file_name in os.listdir(path_to_directory):
            self.download_file_to_local(os.path.join(path_to_directory, file_name), os.path.join(local_path, file_name))

    def write_bytes(self, path_to_file: str, data: bytes) -> None:
        with open(path_to_file, 'wb') as file:
            file.write(data)

    def write_text(self, path_to_file: str, data: str) -> None:
        with open(path_to_file, 'w') as file:
            file.write(data)

    def read_bytes(self, path_to_file: str) -> bytes:
        with open(path_to_file, 'rb') as file:
            return file.read()

    def read_text_file(self, path_to_text_file: str, normalize_special_characters: bool = True) -> str:
        with open(path_to_text_file, 'r') as file:
            text = file.read()
            if normalize_special_characters:
                text = StringUtils.normalize_special_characters(text)
            return text

    def read_jsonl_file(self, path_to_jsonl_file: str) -> list[dict]:
        with open(path_to_jsonl_file, 'r') as file:
            return [json.loads(line) for line in file]

    def read_json_file(self, path_to_json_file: str) -> dict:
        text = self.read_text_file(path_to_json_file, normalize_special_characters=False)
        return json.loads(text)

    def read_text_files_from_dir(self, path_to_local_directory: str, file_extensions: list[str] = ['.txt'], normalize_special_characters: bool = True) -> dict[str, str]:
        contents_by_filename = {}
        if not os.path.isdir(path_to_local_directory):
            raise ValueError(f"{path_to_local_directory} is not a directory")
        for file_name in os.listdir(path_to_local_directory):
            if file_name.endswith(tuple(file_extensions)):
                text = self.read_text_file(os.path.join(path_to_local_directory, file_name), normalize_special_characters)
                contents_by_filename[file_name] = text
        return contents_by_filename

    def read_image(self, local_path_to_image_file: str) -> PILImage.Image:
        return PILImage.open(local_path_to_image_file)

    def read_images_from_dir(self, local_path_to_image_directory: str, file_extensions: list[str] = ['.jpg', '.png', 'jpeg', '.gif', '.bmp']) -> dict[str, PILImage.Image]:
        images_by_filename = {}
        for file_name in os.listdir(local_path_to_image_directory):
            if file_name.endswith(tuple(file_extensions)):
                image = PILImage.open(os.path.join(local_path_to_image_directory, file_name))
                image.info['proton_file_name'] = file_name
                images_by_filename[file_name] = image
        return images_by_filename

    def write_text_file(self, output_file_path: str, text: str) -> None:
        if not output_file_path:
            raise InvalidArgsError('output_file_path must not be empty.')
        with open(output_file_path, 'w') as outfile:
            outfile.write(text)

    def concatinate_text_files(self, source_file_paths: list[str], output_file_path: str) -> None:
        with open(output_file_path, 'w') as outfile:
            for local_path in source_file_paths:
                with open(local_path) as infile:
                    text = infile.read().strip()
                    outfile.write(text)
                if local_path != source_file_paths[-1]:
                    outfile.write('\n')

    def get_file_size(self, path_to_file: str) -> int:
        return os.path.getsize(path_to_file) or 0

    def save_file(self, src_file: str, dest_file: str, overwrite: bool) -> None:
        if overwrite and os.path.isfile(dest_file):
            self.remove_file(dest_file)
        with open(dest_file, 'wb') as file:
            file.write(self.read_bytes(src_file))

    def final_file_uri(self, file_path: str) -> str:
        """ Constructs the final output uri """
        return file_path

    def remove_file(self, path_to_file: str) -> None:
        os.remove(path_to_file)
