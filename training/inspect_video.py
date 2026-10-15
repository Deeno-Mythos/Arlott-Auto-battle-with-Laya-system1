import argparse
import hashlib
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent


def frame_at(container, stream, seconds):
    container.seek(int(seconds / float(stream.time_base)), stream=stream, backward=True)
    for frame in container.decode(stream):
        if frame.time is not None and frame.time >= seconds:
            return frame.to_image()
    raise RuntimeError(f"No frame at {seconds}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    args = parser.parse_args()
    path = Path(args.video)
    out = ROOT / 'data' / 'source'
    out.mkdir(parents=True, exist_ok=True)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration / av.time_base)
        metadata = {'path': str(path.resolve()), 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
                    'duration': duration, 'width': stream.width, 'height': stream.height, 'fps': float(stream.average_rate)}
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        times = [min(duration - 1, i * 30) for i in range(int(duration // 30) + 1)]
        sheet = Image.new('RGB', (4 * 320, ((len(times) + 3) // 4) * 205), '#151924')
        draw = ImageDraw.Draw(sheet)
        for i, seconds in enumerate(times):
            image = frame_at(container, stream, seconds)
            image.save(out / f'frame-{int(seconds):04d}.jpg', quality=92)
            image.thumbnail((320, 180))
            x, y = (i % 4) * 320, (i // 4) * 205
            sheet.paste(image, (x, y + 23))
            draw.text((x + 6, y + 5), f'{seconds:.0f}s', fill='white')
        sheet.save(out / 'timeline.jpg', quality=92)
        print(json.dumps(metadata))
        print(out / 'timeline.jpg')


if __name__ == '__main__':
    main()
