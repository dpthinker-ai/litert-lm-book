"""Page-boundary regressions with fractional DOM and PDF coordinates."""
import unittest
import fitz
from paginate_webkit_pdf import choose_pages, render_pages


class PageBoundaryTests(unittest.TestCase):
    def test_two_snaps_cannot_accumulate_height_tolerance(self):
        # PDF line bottom and DOM heading bottom differ by 0.6 px.
        # Neither candidate is legal on a page with exactly 100 px available.
        metadata = {
            'contentHeight': 100, 'unitStarts': [],
            'keepRanges': [
                {'top': 80, 'bottom': 101.2, 'height': 21.2, 'type': 'heading'},
            ],
        }
        pages, _ = choose_pages(metadata, 180, [40, 80, 100.6, 101.2, 140, 180])
        self.assertEqual(pages[0]['end'], 80)
        for page in pages:
            self.assertLessEqual(page['end'] - page['start'], 100)
            self.assertFalse(80 < page['end'] < 101.2)

    def test_repeated_source_is_printed_in_full_on_each_page(self):
        source = fitz.open()
        page = source.new_page(width=200, height=600)
        page.insert_text((10, 420), "FULL_SOURCE_REFERENCE", fontsize=8)
        metadata = {
            'pageWidth': 200, 'height': 600, 'pageTop': 10, 'pageSide': 10,
            'contentHeight': 200, 'footnoteGap': 5,
            'footnotes': [{'key': 'source', 'top': 400, 'bottom': 430, 'height': 30}],
        }
        pages = [
            {'start': a, 'end': a + 80, 'footnotes': ['source'], 'footnoteHeight': 35}
            for a in (0, 80)
        ]
        output = render_pages(source, metadata, pages)
        for page in output:
            self.assertIn("FULL_SOURCE_REFERENCE", page.get_text())
        output.close()
        source.close()

    def test_footnotes_stay_with_reference_and_within_page(self):
        metadata = {
            'contentHeight': 100, 'unitStarts': [], 'keepRanges': [],
            'footnoteGap': 5, 'maxFootnoteHeight': 35,
            'footnoteRefs': [{'key': 'source', 'y': 50}],
            'footnotes': [{'key': 'source', 'height': 20}],
        }
        pages, _ = choose_pages(metadata, 180, [20, 40, 60, 75.5, 100, 140, 180])
        with_note = [p for p in pages if p['footnotes']]
        self.assertEqual(len(with_note), 1)
        self.assertTrue(with_note[0]['start'] <= 50 < with_note[0]['end'])
        for page in pages:
            self.assertLessEqual(page['end'] - page['start'] + page['footnoteHeight'], 100)


if __name__ == '__main__':
    unittest.main()
