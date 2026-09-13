"""Regressions for duplicate WebKit heading text and incomplete PDF outlines."""
import re
import unittest
from unittest.mock import patch
import finalize_pdf as pdf


class Page:
    def __init__(self, lines):
        self.lines=lines

    def get_text(self, mode):
        return {'blocks':[{'type':0,'lines':[
            {'spans':[{'text':text,'size':size}]} for text,size in self.lines]}]}


class OutlineTests(unittest.TestCase):
    def test_partial_glyph_line_does_not_hide_complete_heading(self):
        # Observed on raw page 164: three extraction lines from one visual heading.
        page=Page([('11 5',16.94),('按数据边界定位多模态输⼊失败',16.94),
                   ('11.5\u3000按数据边界定位多模态输⼊失败',16.94)])
        candidates=pdf.heading_blocks(page,9.97)
        self.assertTrue(any(re.match(r'^11\.5(?!\d)(?!\.\d)',s) for s in candidates))

    def test_body_reference_in_same_block_is_not_a_heading(self):
        page=Page([('11.4 标题',16.94),('11.5 正文中的前向引用',9.97)])
        candidates=pdf.heading_blocks(page,9.97)
        self.assertFalse(any(s.startswith('11.5') for s in candidates))

    def test_missing_section_stops_export(self):
        class Doc:
            page_count=1
            def __getitem__(self,index):return Page([('普通正文',10)])
        chapter=next(i for i,unit in enumerate(pdf.UNITS)
                     if re.search(r'第\s*1\s*章',unit[0]))
        with patch.object(pdf,'load_section_titles',return_value={1:[('1.1','1.1 必须存在')]}):
            with self.assertRaisesRegex(RuntimeError,'1.1 必须存在'):
                pdf.find_section_pages(Doc(),{chapter:(0,0)},{chapter:chapter},[0])


if __name__=='__main__':unittest.main()
