import array
import json
import base64
import struct

class Serialization():
    @staticmethod
    def json_to_b85(obj):
        return base64.b85encode(json.dumps(obj).encode()).decode()

    @staticmethod
    def b85_to_json(encoded: str):
        return json.loads(base64.b85decode(encoded).decode())

    @staticmethod
    def float_list_to_b85(float_list: list[float]) -> str:
        buffer = struct.pack(f'{len(float_list)}f', *float_list)
        # buffer = struct.pack('!I' + 'd' * len(arr), len(arr), *arr)
        return base64.b85encode(buffer).decode()

    @staticmethod
    def b85_to_float_list(encoded: str) -> list[float]:
        buffer = base64.b85decode(encoded)
        float_arr = array.array('f')
        float_arr.frombytes(buffer)
        return list(float_arr)