import unittest
from m4_summarize import callback_text

class ConversationCallbackTests(unittest.TestCase):
    def test_json_metadata_is_not_visible_text(self):
        self.assertEqual(callback_text('{"role":"assistant","content":[]}'),'')
        self.assertEqual(callback_text('{"role":"assistant","tool_calls":[{}]}'),'')
    def test_only_text_parts_are_combined(self):
        self.assertEqual(callback_text('{"content":[{"type":"image","text":"not output"},{"type":"text","text":"red"},{"type":"text","text":" square"}]}'),'red square')
    def test_empty_final_and_string_content(self):
        self.assertEqual(callback_text(''),'')
        self.assertEqual(callback_text('{"content":"你好"}'),'你好')
    def test_malformed_callback_is_not_silently_scored(self):
        with self.assertRaises(ValueError):callback_text('not JSON')

if __name__=='__main__':unittest.main()
