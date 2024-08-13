import re

class NumberUtils():
    @staticmethod
    def int_to_roman( number : int) -> str:
        map = [(10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
        result = []
        for (arabic, roman) in map:
            (factor, number) = divmod(number, arabic)
            result.append(roman * factor)
            if number == 0:
                break
        return ''.join(result)

    @staticmethod
    def get_ordinal_suffix(number: int) -> str:
        return ('th' if 4<=number%100<=20 else {1:'st',2:'nd',3:'rd'}.get(number%10, 'th'))

    @staticmethod
    def parse_float(text: str) -> float:
        text = text.replace('$', '').strip()
        text = text.replace('%', '').strip()
        if not text:
            raise ValueError(f'parse_float called with empty string')
        return float(text)

    @staticmethod
    def parse_integers(text: str) -> list[int]:
        """Extract all integer numbers from the given string, for example '1,2,3,000.99' -> [1,2,3,0,99]"""
        numeric_strings = re.findall(r'\d+', text)
        return [int(s) for s in numeric_strings]

    # function to parse multiple float numbers from a given string
    @staticmethod
    def parse_floats(text: str) -> list[float]:
        """Extract all numbers from the given string, for example '1+2=3.14' -> [1,2,3.14]"""
        numeric_strings = re.findall(r'[-+]?\d*\.?\d+', text)
        return [float(s) for s in numeric_strings]


    @staticmethod
    def split_array_into_contiguous_ranges(sorted_array: list[int]) -> list[list[int]]:
        ranges = []
        range = []
        previous = -1
        for current in sorted_array:
            if previous+1 != current:            # end of previous subList and beginning of next
                if range:              # if subList already has elements
                    ranges.append(range)
                    range = []
            range.append(current)
            previous = current
        if range:
            ranges.append(range)
        return ranges

    @staticmethod
    def round_float(value: float) -> float:
        if value > 100:
            return round(value, 1)
        elif value > 10:
            return round(value, 2)
        elif value > 1:
            return round(value, 3)
        else:
            return round(value, 4)