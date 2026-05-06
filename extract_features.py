"""
Extracao de caracteristicas - Trabalho Inteligencia Computacional
Bases: OCR (digitos) e Meses do Ano
Saida: 6 arquivos .txt (handcrafted e non-handcrafted x treino/teste/meses)
"""
import os
import re
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

BASE = os.path.dirname(os.path.abspath(__file__))
OCR_TREINO_DIR = os.path.join(BASE, "treino")
OCR_TESTE_DIR  = os.path.join(BASE, "teste")
MESES_DIR      = os.path.join(BASE, "Base de Dados - Meses do Ano")

OCR_SIZE   = (32, 32)
MESES_SIZE = (128, 32)
GRID       = (8, 8)

MES_LABEL = {
    "j": 0, "f": 1, "m": 2, "a": 3,
    "md": 4, "jd": 5, "jt": 6, "ad": 7,
    "s": 8, "o": 9, "n": 10, "d": 11,
}
MES_PREFIXES_SORTED = sorted(MES_LABEL.keys(), key=len, reverse=True)


def preprocess(path, target_size):
    """Le imagem, normaliza tinta=1/fundo=0, redimensiona com NEAREST."""
    arr = np.array(Image.open(path).convert("L"))
    if (arr == 255).sum() < (arr == 0).sum():
        fg = (arr == 255).astype(np.uint8)
    else:
        fg = (arr == 0).astype(np.uint8)
    img = Image.fromarray(fg * 255)
    img = img.resize(target_size, Image.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def zoning(fg, grid=GRID):
    """Fracao de pixels de tinta por celula da grade. Retorna vetor grid[0]*grid[1]."""
    H, W = fg.shape
    rows, cols = grid
    cell_h, cell_w = H // rows, W // cols
    feats = np.empty(rows * cols, dtype=np.float64)
    k = 0
    for i in range(rows):
        for j in range(cols):
            block = fg[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            feats[k] = block.mean()
            k += 1
    return feats


def list_ocr(folder):
    items = []
    for cls in sorted(os.listdir(folder)):
        cls_dir = os.path.join(folder, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if f.lower().endswith(".bmp"):
                items.append((os.path.join(cls_dir, f), int(cls)))
    return items


def label_from_meses_name(filename):
    name = os.path.basename(filename)
    for pref in MES_PREFIXES_SORTED:
        if name.startswith(pref) and name[len(pref):len(pref)+1].isdigit():
            return MES_LABEL[pref]
    raise ValueError(f"Prefixo nao reconhecido: {name}")


def list_meses(folder):
    items = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".bmp"):
            items.append((os.path.join(folder, f), label_from_meses_name(f)))
    return items


def build_resnet():
    """ResNet18 pre-treinada com a camada final removida -> features de 512 dims."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model


RESNET_TFM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def resnet_features(model, paths, batch_size=64):
    """Extrai vetores de 512-d para uma lista de caminhos. Usa batches."""
    feats_all = np.empty((len(paths), 512), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start+batch_size]
            tensors = []
            for p in chunk:
                arr = np.array(Image.open(p).convert("L"))
                if (arr == 255).sum() < (arr == 0).sum():
                    fg = (arr == 255).astype(np.uint8) * 255
                else:
                    fg = (arr == 0).astype(np.uint8) * 255
                rgb = np.stack([fg, fg, fg], axis=-1)
                pil = Image.fromarray(rgb)
                tensors.append(RESNET_TFM(pil))
            batch = torch.stack(tensors, dim=0)
            out = model(batch).cpu().numpy()
            feats_all[start:start+len(chunk)] = out
            print(f"    [resnet] {start+len(chunk)}/{len(paths)}")
    return feats_all


def write_handcrafted(items, target_size, out_path):
    print(f"  -> {out_path}  (n={len(items)})")
    rows = np.empty((len(items), GRID[0]*GRID[1] + 1), dtype=np.float64)
    for i, (path, label) in enumerate(items):
        fg = preprocess(path, target_size)
        rows[i, :-1] = zoning(fg)
        rows[i, -1] = label
    fmt = ["%.6f"] * (GRID[0]*GRID[1]) + ["%d"]
    np.savetxt(out_path, rows, fmt=fmt)


def write_resnet(items, model, out_path):
    print(f"  -> {out_path}  (n={len(items)})")
    paths = [p for p, _ in items]
    labels = np.array([l for _, l in items], dtype=np.int64).reshape(-1, 1)
    feats = resnet_features(model, paths)
    rows = np.hstack([feats.astype(np.float64), labels])
    fmt = ["%.6f"] * 512 + ["%d"]
    np.savetxt(out_path, rows, fmt=fmt)


def main():
    print("Listando bases...")
    ocr_treino = list_ocr(OCR_TREINO_DIR)
    ocr_teste  = list_ocr(OCR_TESTE_DIR)
    meses      = list_meses(MESES_DIR)
    print(f"  OCR treino: {len(ocr_treino)} imagens")
    print(f"  OCR teste : {len(ocr_teste)} imagens")
    print(f"  Meses     : {len(meses)} imagens")

    print("\n[1/2] Handcrafted (zoneamento 8x8)")
    write_handcrafted(ocr_treino, OCR_SIZE,   os.path.join(BASE, "ocr_handcrafted_treino.txt"))
    write_handcrafted(ocr_teste,  OCR_SIZE,   os.path.join(BASE, "ocr_handcrafted_teste.txt"))
    write_handcrafted(meses,      MESES_SIZE, os.path.join(BASE, "meses_handcrafted.txt"))

    print("\n[2/2] Non-handcrafted (ResNet18)")
    model = build_resnet()
    write_resnet(ocr_treino, model, os.path.join(BASE, "ocr_resnet_treino.txt"))
    write_resnet(ocr_teste,  model, os.path.join(BASE, "ocr_resnet_teste.txt"))
    write_resnet(meses,      model, os.path.join(BASE, "meses_resnet.txt"))

    print("\nConcluido. 6 arquivos gerados na pasta:", BASE)


if __name__ == "__main__":
    main()
