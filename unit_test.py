import unittest

class TestStringMethods(unittest.TestCase):
    def fix_line_breaks(self, text: str) -> str:
        print(text.splitlines())
        text = '\n'.join(text.splitlines())
        text = text.replace('\r', '')
        print(text)
        return text
    
    def test_split(self):
        original_text = "First line.\nSecond line.\rThird line."
        goal_text = "First line.\nSecond line.\nThird line."
        test_text = self.fix_line_breaks(original_text)
        self.assertEqual(test_text, goal_text)

if __name__ == '__main__':
    unittest.main()