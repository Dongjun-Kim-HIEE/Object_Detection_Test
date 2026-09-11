"""Prepare remapped coin/battery labels and fine-tune YOLO11n."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
NAMES = ['Coin', 'Pouch battery', 'battery', 'cylindrical battery', 'drycell', 'prismatic battery']


def prepare():
    output = ROOT / 'combined_dataset'
    counts = {}
    for folder, prefix, offset, classes in [('coin.v1i.yolov11', 'coin', 0, 1), ('battery.yolov11', 'battery', 1, 5)]:
        for split in ['train', 'valid', 'test']:
            images = ROOT / folder / split / 'images'
            count = 0
            for src in sorted(images.glob('*')):
                if src.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    continue
                label = src.parent.parent / 'labels' / (src.stem + '.txt')
                if not label.exists():
                    raise ValueError(f'Missing label: {label}')
                lines = []
                for line in label.read_text().splitlines():
                    if not line.strip():
                        continue
                    values = line.split()
                    if len(values) >= 7 and len(values) % 2 == 1:
                        points = list(map(float, values[1:]))
                        if not all(math.isfinite(x) and 0 <= x <= 1 for x in points):
                            raise ValueError(f'Invalid polygon: {label}')
                        xs, ys = points[0::2], points[1::2]
                        values = [values[0], str((min(xs)+max(xs))/2), str((min(ys)+max(ys))/2), str(max(xs)-min(xs)), str(max(ys)-min(ys))]
                    if len(values) != 5:
                        raise ValueError(f'Invalid detection label: {label}: {line}')
                    cls = int(values[0])
                    coords = list(map(float, values[1:]))
                    if not 0 <= cls < classes or not all(math.isfinite(x) and 0 <= x <= 1 for x in coords) or min(coords[2:]) <= 0:
                        raise ValueError(f'Invalid class/coordinates: {label}: {line}')
                    lines.append(' '.join([str(cls + offset), *values[1:]]))
                dst = output / split / 'images' / f'{prefix}_{src.name}'
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                target = output / split / 'labels' / f'{prefix}_{src.stem}.txt'
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
                count += 1
            counts[f'{prefix}/{split}'] = count
    config = output / 'data.yaml'
    config.write_text(yaml.safe_dump({'path': output.as_posix(), 'train': 'train/images', 'val': 'valid/images', 'names': NAMES}, allow_unicode=True), encoding='utf-8')
    print(json.dumps(counts), flush=True)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = prepare()
    if args.prepare_only:
        return
    if args.resume:
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        model = YOLO(str(ROOT / 'yolo11n.pt'))
        model.train(data=str(config), epochs=args.epochs, imgsz=640, batch=8,
                    device='cpu', workers=0, project=str(ROOT / 'runs'),
                    name='coins_battery', patience=10, seed=42, cache=False)
    best = Path(model.trainer.best)
    validated = YOLO(str(best))
    metrics = validated.val(data=str(config), device='cpu', workers=0)
    (best.parent.parent / 'validation.json').write_text(json.dumps(metrics.results_dict, indent=2), encoding='utf-8')
    (ROOT / 'models').mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / 'models' / 'coins_battery.pending.pt')
    (ROOT / 'models' / 'coins_battery.pending.pt').replace(ROOT / 'models' / 'coins_battery.pt')
    print('Training complete. Custom weights installed.', flush=True)


if __name__ == '__main__':
    main()
