#!/usr/bin/env python3
"""Read bounded header ranges from pinned public MoE artifacts; never run them."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from litert_lm_builder import litertlm_peek as peek

REPO = 'litert-community/gemma-4-26B-A4B-it-litert-lm'
REVISION = '7228819fa9580751b57b41a93ee54d5c08c4e001'
HEADER_BYTES = 32768


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    api = f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true'
    with urlopen(api, timeout=30) as response:
        info = json.load(response)
    assert info['sha'] == REVISION
    report = dict(fetched_at_utc=datetime.now(timezone.utc).isoformat(), repo=REPO,
        revision=REVISION, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        builder_version=importlib.metadata.version('litert-lm-builder'), files=[],
        limitations=['Only first 32768 bytes read; no full model download, hash verification or inference.',
                     'Full-file hashes and sizes are publisher metadata, not local verification.'])
    for variant in ('gpu', 'web'):
        name = f'gemma-4-26B-A4B-it-{variant}.litertlm'
        published = next(s for s in info['siblings'] if s['rfilename'] == name)
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        request = Request(url, headers={'Range': f'bytes=0-{HEADER_BYTES-1}', 'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=60) as response:
            expected_range = f'bytes 0-{HEADER_BYTES-1}/{published["size"]}'
            assert response.status == 206 and response.headers['Content-Range'] == expected_range
            data = response.read(HEADER_BYTES+1)
            assert len(data) == HEADER_BYTES
        path = args.output/(name+'.header')
        path.write_bytes(data)
        version = list(struct.unpack_from('<III', data, 8))
        metadata = peek.read_litertlm_header(str(path), io.StringIO())
        sections = metadata.SectionMetadata()
        entries = []
        for index in range(sections.ObjectsLength()):
            section = sections.Objects(index)
            items = {}
            for item_index in range(section.ItemsLength()):
                item = section.Items(item_index)
                items[item.Key().decode()] = peek._get_kvp_value_and_type(item)[0]
            entries.append(dict(index=index, begin=section.BeginOffset(), end=section.EndOffset(),
                                data_type=section.DataType(), items=items))
        report['files'].append(dict(name=name, url=url, published_size=published['size'],
            published_lfs_sha256=published['lfs']['sha256'], http_status=206,
            content_range=expected_range, header_file=path.name,
            header_sha256=hashlib.sha256(data).hexdigest(), container_version=version, sections=entries))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
