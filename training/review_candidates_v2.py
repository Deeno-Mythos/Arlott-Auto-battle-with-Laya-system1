"""Generate timestamped evidence for label audit; source footage remains unchanged."""
import json
import argparse
from pathlib import Path
import av
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'training/data';OUT=DATA/'v2/review'
WINDOWS=[58,72,90,118,128,145,197,267,297,326,409,467,492,537,607,694,716,758,764,766]


def main():
    global WINDOWS,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--windows',nargs='+',type=float)
    parser.add_argument('--tag',default='review');args=parser.parse_args()
    if args.windows:WINDOWS=args.windows
    OUT=DATA/'v2'/args.tag
    OUT.mkdir(parents=True,exist_ok=True)
    video=json.loads((DATA/'source/metadata.json').read_text())['path']
    for page in range((len(WINDOWS)+4)//5):
        sheet=Image.new('RGB',(1440,5*185),(15,19,27));draw=ImageDraw.Draw(sheet)
        with av.open(video) as container:
            stream=container.streams.video[0]
            for row,start in enumerate(WINDOWS[page*5:page*5+5]):
                for col,offset in enumerate((0,.5,1,1.5,2)):
                    target=start+offset;container.seek(int(target/stream.time_base),stream=stream,backward=True)
                    frame=next(f for f in container.decode(stream) if f.time is not None and f.time>=target)
                    img=frame.to_image();img.save(OUT/f'{target:07.2f}.jpg')
                    img.thumbnail((288,162));sheet.paste(img,(col*288,row*185+21))
                    draw.text((col*288+4,row*185+3),f'{frame.time:.2f}s',fill='white')
        sheet.save(OUT/f'page-{page}.jpg')
    print(str(OUT))


if __name__=='__main__':main()
