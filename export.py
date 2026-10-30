import argparse, base64, inspect, io, json, os, pickle
import numpy as np
import torch
from Autoencoder import ConvAutoencoder
from util import *

GAMES = ['smb', 'ki', 'mm', 'met']
DIRS = {'smb': 'SMB', 'ki': 'KI', 'mm': 'MM', 'met': 'Met'}

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='web_assets', type=str)
parser.add_argument('--z', default=256, type=int)
parser.add_argument('--epochs', default=250, type=int)
parser.add_argument('--verify', default=1, type=int)
args = parser.parse_args()

def write_json(name, obj):
    with open(os.path.join(args.out, name), 'w') as f:
        json.dump(obj, f, separators=(',', ':'))

def export_tiles():
    uris = {}
    for char, img in images.items():
        buf = io.BytesIO()
        img.convert('RGBA').save(buf, 'PNG')
        uris[char] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    write_json('tiles.json', uris)

def export_levels():
    lib = {}
    for game in GAMES:
        vglc = f'VGLC/{DIRS[game]}/'
        levels = {}
        for f in natural_sort(os.listdir(vglc)):
            level = open(vglc + f).read().splitlines()
            levels[f[:-4]] = preprocess_level(level, game)
        chunks, _ = parse_folder(f'data/{game_folders_nor[game]}', game)
        # lib[game] = {'levels': levels, 'chunks': [[''.join(row) for row in c] for c in chunks]}
        lib[game] = levels
    write_json('levels.json', lib)

def export_mrf():
    os.makedirs(os.path.join(args.out, 'mrf'), exist_ok=True)
    for game in GAMES:
        for ns in (4, 8):
            with open(f'trained_mrf/mrf_{ns}_{game}.pickle', 'rb') as f:
                write_json(f'mrf/mrf_{ns}_{game}.json', pickle.load(f))

def export_ae():
    os.makedirs(os.path.join(args.out, 'models'), exist_ok=True)
    extra = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
    for game in GAMES:
        model = ConvAutoencoder(len(sketch_chars), len(int2chars[game]), args.z)
        model.load_state_dict(torch.load(f'trained_ae/ae_full_conv_{game}_ld_{args.z}_{args.epochs}.pth',
                                         map_location='cpu'))
        model.eval()
        path = os.path.join(args.out, 'models', f'ae_conv_{game}.onnx')
        torch.onnx.export(model, torch.zeros(1, len(sketch_chars), 15, 16), path,
                          input_names=['sketch'], output_names=['logits'],
                          dynamic_axes={'sketch': {0: 'batch'}, 'logits': {0: 'batch'}},
                          opset_version=17, **extra)
        print('Exported', path)
        if args.verify:
            verify(model, path, game)


def verify(model, onnx_path, game, n=20):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    src = next(g for g in GAMES if g != game)
    chunks, _ = parse_folder(f'data/{game_folders_nor[src]}', src)
    chunks = chunks[:n]
    bad = 0
    with torch.no_grad():
        for chunk in chunks:
            sketch = translate_level([''.join(row) for row in chunk], src)
            expected = apply_ae(model, sketch, len(int2chars[game]), game, 'conv')
            idx = np.array([[sc2int[c] for c in row] for row in sketch])
            x = np.eye(len(sketch_chars), dtype=np.float32)[idx].transpose(2, 0, 1)[None]
            logits = sess.run(None, {'sketch': x})[0]
            got = [''.join(int2chars[game][t] for t in row) for row in logits.argmax(1)[0]]
            bad += got != expected
    print(f'  {game}: {len(chunks) - bad}/{len(chunks)} {src} chunks match PyTorch')

os.makedirs(args.out, exist_ok=True)
export_tiles()
export_levels()
export_mrf()
write_json('meta.json', {'int2chars': int2chars, 'aff_to_tile': aff_to_tile})
export_ae()
print('Wrote assets to', args.out)