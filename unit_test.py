import unittest

class TestStringMethods(unittest.TestCase):
    def fix_line_breaks(self, text: str) -> str:
        text = '\n'.join(text.splitlines())
        text = text.replace('\r', '')
        return text
    
    def test_split(self):
        text = "First line.\nSecond line.\rThird line."
        text1 = self.fix_line_breaks(text)
        goal_text = "First line.\nSecond line.\nThird line."
        
        self.assertEqual(text1, goal_text)

if __name__ == '__main__':
    unittest.main()