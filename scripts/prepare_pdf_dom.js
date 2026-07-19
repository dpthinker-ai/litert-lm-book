const PAGE_WIDTH = 794;
const PX_PER_MM = PAGE_WIDTH / 210;
const PAGE_HEIGHT = PAGE_WIDTH * 297 / 210;
const PAGE_TOP = 16 * PX_PER_MM;
const PAGE_SIDE = 17 * PX_PER_MM;
const PAGE_BOTTOM = 22 * PX_PER_MM;
const CONTENT_HEIGHT = PAGE_HEIGHT - PAGE_TOP - PAGE_BOTTOM;
const FOOTNOTE_GAP = 14;
const MAX_FOOTNOTE_HEIGHT = CONTENT_HEIGHT * 0.35;

// The renderer keeps its window off-screen, where requestAnimationFrame may be
// suspended indefinitely. A short timer lets WebKit finish style and layout.
const nextTwoFrames = () => new Promise(resolve => setTimeout(resolve, 100));

document.documentElement.classList.remove(
  'sidebar-visible', 'navy', 'coal', 'ayu', 'rust'
);
document.documentElement.classList.add('light');

// WKWebView.createPDF() uses screen media. Apply both mdBook's standalone
// print stylesheet and the @media print rules in assets/book.css.
for (const link of document.querySelectorAll(
  'link[rel~="stylesheet"][media]'
)) {
  if (/\bprint\b/i.test(link.media)) link.media = 'all';
}
await nextTwoFrames();

const printRules = [];
for (const sheet of document.styleSheets) {
  try {
    for (const rule of sheet.cssRules) {
      if (rule.type !== CSSRule.MEDIA_RULE ||
          !/\bprint\b/i.test(rule.conditionText)) {
        continue;
      }
      for (const inner of rule.cssRules) {
        if (!/^\s*@page\b/i.test(inner.cssText)) {
          printRules.push(inner.cssText);
        }
      }
    }
  } catch (_) {
    // Local book assets are same-origin. Keep an optional external stylesheet
    // from aborting the render.
  }
}

const style = document.createElement('style');
style.id = 'webkit-pdf-rules';
style.textContent = `${printRules.join('\n')}
  html, body {
    width: ${PAGE_WIDTH}px !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
    background: #fff !important;
    color: #1f2328 !important;
  }
  #mdbook-help-container, #mdbook-sidebar, #mdbook-menu-bar,
  #mdbook-menu-bar-hover-placeholder, #mdbook-search-wrapper,
  #mdbook-searchresults-outer, #mdbook-sidebar-toggle-anchor,
  #mdbook-content > .nav-wrapper, .nav-wide-wrapper,
  .nav-chapters, .mobile-nav-chapters, pre > .buttons {
    display: none !important;
  }
  #mdbook-page-wrapper.page-wrapper, .page, #mdbook-content.content {
    width: ${PAGE_WIDTH}px !important;
    min-width: 0 !important;
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
    transform: none !important;
    overflow: visible !important;
    background: #fff !important;
  }
  #mdbook-content > main {
    box-sizing: border-box !important;
    width: ${PAGE_WIDTH - 2 * PAGE_SIDE}px !important;
    max-width: ${PAGE_WIDTH - 2 * PAGE_SIDE}px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: #fff !important;
  }
  :root {
    --fig-bg:#fff; --fig-surface:#f6f8fa; --fig-surface-2:#eef1f5;
    --fig-accent:#0b57d0; --fig-accent-2:#6639ba;
    --fig-border:#d0d7de; --fig-border-2:#e6eaef;
    --fig-text:#1f2328; --fig-text-soft:#57606a;
    --fig-text-dim:#8b949e; --fig-green:#1a7f37;
  }
  pre, pre code { font-size: 12px; line-height: 1.5; }
  pre code { white-space: pre-wrap !important; overflow-wrap: anywhere; }
  .table-wrapper { overflow: visible !important; }
  table { width: 100% !important; font-size: 10px !important; line-height: 1.45; }
  th, td { word-break: break-word; padding: 4px 7px !important; }
  table code { white-space: normal !important; word-break: break-all; font-size: 9px; }
  .book-cover { padding-top: 90px !important; }
`;
document.head.append(style);
await nextTwoFrames();

const main = document.querySelector('#mdbook-content > main');
if (!main) throw new Error('missing #mdbook-content > main');

const rectOf = element => {
  const rect = element.getBoundingClientRect();
  return {
    top: rect.top + scrollY,
    bottom: rect.bottom + scrollY,
    height: rect.height,
  };
};

// mdBook renders semantic footnotes at the end of each source page. The print
// view concatenates all source pages, so first validate that their generated
// ids remain unique. Then remove the definitions from the body flow and clone
// them into a vector bank below the body. The paginator will place bank slices
// at the bottom of the physical pages that contain their references.
const definitionItems = [
  ...main.querySelectorAll('ol.footnote-definition > li[id]'),
];
const definitions = new Map();
for (const item of definitionItems) {
  if (definitions.has(item.id)) {
    throw new Error(`duplicate footnote definition: ${item.id}`);
  }
  definitions.set(item.id, item);
}

const referenceElements = [
  ...main.querySelectorAll('sup.footnote-reference'),
];
const references = [];
const referencedKeys = new Set();
for (const reference of referenceElements) {
  if (reference.closest('ol.footnote-definition')) {
    throw new Error('footnote references inside footnote definitions are unsupported');
  }
  const link = reference.querySelector(':scope > a[href^="#footnote-"]');
  if (!link) throw new Error('footnote reference is missing its definition link');
  const key = link.getAttribute('href').slice(1);
  if (!definitions.has(key)) {
    throw new Error(`unknown footnote definition: ${key}`);
  }
  referencedKeys.add(key);
  references.push({
    element: reference,
    key,
    number: link.textContent.trim(),
  });
}
for (const key of definitions.keys()) {
  if (!referencedKeys.has(key)) {
    throw new Error(`unreferenced footnote definition: ${key}`);
  }
}

for (const list of new Set(definitionItems.map(item => item.parentElement))) {
  const separator = list.previousElementSibling;
  if (separator?.tagName === 'HR') separator.remove();
  list.remove();
}
await nextTwoFrames();

const bodyChildren = [...main.children];
const bodyBottom = Math.max(
  0,
  ...bodyChildren
    .filter(element => getComputedStyle(element).display !== 'none')
    .map(element => rectOf(element).bottom),
);

let footnoteBank = null;
if (definitions.size > 0) {
  footnoteBank = document.createElement('section');
  footnoteBank.className = 'pdf-footnote-bank';
  footnoteBank.setAttribute('aria-label', 'PDF footnote source bank');

  for (const [key, item] of definitions) {
    const firstReference = references.find(reference => reference.key === key);
    const bankItem = document.createElement('div');
    bankItem.className = 'pdf-footnote-bank-item';
    bankItem.dataset.footnoteKey = key;

    const label = document.createElement('span');
    label.className = 'pdf-footnote-bank-label';
    label.textContent = `${firstReference.number}.`;

    const content = document.createElement('div');
    content.className = 'pdf-footnote-bank-content';
    const clone = item.cloneNode(true);
    clone.removeAttribute('id');
    for (const element of clone.querySelectorAll('[id]')) {
      element.removeAttribute('id');
    }
    for (const backlink of clone.querySelectorAll('a[href^="#fr-"]')) {
      backlink.remove();
    }
    while (clone.firstChild) content.append(clone.firstChild);

    bankItem.append(label, content);
    footnoteBank.append(bankItem);
  }
  main.append(footnoteBank);
  await nextTwoFrames();
}

const unitMarkers = [...main.children].filter(element =>
  element.tagName === 'DIV' &&
  (element.style.breakBefore === 'page' ||
   element.style.pageBreakBefore === 'always')
);
const unitStarts = unitMarkers.map(marker => {
  const next = marker.nextElementSibling;
  const measured = rectOf(next);
  return {
    y: measured.top,
    id: next.id || '',
    text: next.textContent.trim(),
  };
});

const keepRanges = [];
function addRange(element, type, bottom = null) {
  const measured = rectOf(element);
  const end = bottom === null ? measured.bottom : bottom;
  if (measured.height <= 0.5 || end <= measured.top + 0.5) return;
  keepRanges.push({
    top: measured.top,
    bottom: end,
    height: end - measured.top,
    type,
    id: element.id || '',
    text: element.textContent.trim().slice(0, 96),
  });
}

for (const element of main.children) {
  if (!(element instanceof HTMLElement) ||
      getComputedStyle(element).display === 'none' ||
      unitMarkers.includes(element)) {
    continue;
  }

  if (element.matches('h1, h2, h3, h4')) {
    const next = element.nextElementSibling;
    const own = rectOf(element);
    let bottom = own.bottom;
    if (next && !unitMarkers.includes(next)) {
      const following = rectOf(next);
      bottom = Math.min(
        following.bottom,
        own.top + Math.min(CONTENT_HEIGHT, own.height + 96)
      );
    }
    addRange(element, 'heading', bottom);
  } else if (element.classList.contains('table-wrapper')) {
    const tableRect = rectOf(element);
    const hasFootnotes = element.querySelector('.footnote-reference') !== null;
    if (tableRect.height <= CONTENT_HEIGHT * 0.6 && !hasFootnotes) {
      addRange(element, 'table');
    } else {
      for (const row of element.querySelectorAll('tr')) addRange(row, 'row');
    }
  } else if (element.matches('ul, ol')) {
    const listRect = rectOf(element);
    const hasFootnotes = element.querySelector('.footnote-reference') !== null;
    if (listRect.height <= CONTENT_HEIGHT * 0.6 && !hasFootnotes) {
      addRange(element, 'list');
    } else {
      for (const item of element.querySelectorAll(':scope > li')) {
        addRange(item, 'list-item');
      }
    }
  } else if (element.matches(
    'p, figure, pre, blockquote, .aside-compare, .aside-version, .book-cover'
  )) {
    addRange(element, element.tagName.toLowerCase());
  }
}

keepRanges.sort((a, b) => a.top - b.top || b.bottom - a.bottom);
const oversized = keepRanges.filter(range =>
  range.height > CONTENT_HEIGHT + 0.5
);

const footnoteRefs = references.map(reference => ({
  y: rectOf(reference.element).top,
  key: reference.key,
  number: reference.number,
}));
const footnotes = footnoteBank === null ? [] : [
  ...footnoteBank.querySelectorAll(':scope > .pdf-footnote-bank-item'),
].map(item => {
  const measured = rectOf(item);
  if (getComputedStyle(item).display !== 'grid') {
    throw new Error('PDF footnote bank styles were not applied');
  }
  if (measured.height + FOOTNOTE_GAP > MAX_FOOTNOTE_HEIGHT) {
    throw new Error(`footnote is too tall: ${item.dataset.footnoteKey}`);
  }
  return {
    key: item.dataset.footnoteKey,
    top: measured.top,
    bottom: measured.bottom,
    height: measured.height,
  };
});

await nextTwoFrames();
return {
  height: document.documentElement.scrollHeight,
  pageWidth: PAGE_WIDTH,
  pageHeight: PAGE_HEIGHT,
  pageTop: PAGE_TOP,
  pageSide: PAGE_SIDE,
  pageBottom: PAGE_BOTTOM,
  contentHeight: CONTENT_HEIGHT,
  bodyBottom,
  footnoteGap: FOOTNOTE_GAP,
  maxFootnoteHeight: MAX_FOOTNOTE_HEIGHT,
  footnoteRefs,
  footnotes,
  mathContainers: document.querySelectorAll('mjx-container').length,
  unitStarts,
  keepRanges,
  oversized,
};
