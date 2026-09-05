#!/usr/bin/env python3
"""Generate a synthetic, public-content-only RGB test chart; no photographs."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
out=Path('experiments/fixtures/m4'); out.mkdir(parents=True,exist_ok=True)
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480" role="img" aria-label="从左到右为红色正方形、蓝色圆形和绿色三角形">
<defs><style>.bg{fill:var(--fixture-bg,#ffffff)}.red{fill:var(--fixture-red,#dd2020)}.blue{fill:var(--fixture-blue,#205ddd)}.green{fill:var(--fixture-green,#169447)}</style></defs>
<path class="bg" d="M0 0H640V480H0Z"/>
<rect class="red" x="55" y="180" width="120" height="120"/>
<circle class="blue" cx="320" cy="240" r="65"/>
<path class="green" d="M525 170L600 310H450Z"/>
</svg>
'''
(out/'shapes.svg').write_text(svg)
im=Image.new('RGB',(640,480),'white'); d=ImageDraw.Draw(im)
d.rectangle((55,180,174,299),fill='#dd2020');d.ellipse((255,175,385,305),fill='#205ddd');d.polygon([(525,170),(600,310),(450,310)],fill='#169447')
im.save(out/'shapes.png')
(out/'invalid.png').write_bytes(b'This is deliberately not an encoded image.\n')
meta={'kind':'synthetic geometric chart','width':640,'height':480,'mode':'RGB','creator':'book experiment fixture','expected_left_to_right':['red square','blue circle','green triangle'],'pixel_generator':'Pillow; boundaries can differ by one pixel from SVG preview','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.suffix in ('.png','.svg')}}
(out/'fixture.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
