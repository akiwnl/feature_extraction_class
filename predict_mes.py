"""
Classificador de Meses do Ano - KNN
Lê os arquivos gerados pelo extract_features.py e prevê o mês
de uma imagem enviada pelo usuário.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# IMPORTAÇÃO DOS DADOS ORIGINAIS DO HANDCRAFTED
from handcrafted import extract_one, GRID_ROWS, GRID_COLS, MESES_SIZE

BASE = os.path.dirname(os.path.abspath(__file__))

MESES_HANDCRAFTED = os.path.join(BASE, "meses_handcrafted.txt")
MESES_RESNET      = os.path.join(BASE, "meses_resnet.txt")

# Define a tupla GRID baseada nos valores reais do arquivo handcrafted.py
GRID = (GRID_ROWS, GRID_COLS)

# Mesmo mapeamento do extract_features.py
MES_LABEL = {
    "j": 0, "f": 1, "m": 2, "a": 3,
    "md": 4, "jd": 5, "jt": 6, "ad": 7,
    "s": 8, "o": 9, "n": 10, "d": 11,
}

LABEL_TO_MES = {
    0: "Janeiro",
    1: "Fevereiro",
    2: "Marco",
    3: "Abril",
    4: "Maio",
    5: "Junho",
    6: "Julho",
    7: "Agosto",
    8: "Setembro",
    9: "Outubro",
    10: "Novembro",
    11: "Dezembro",
}

# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def header(t):
    b = "=" * 52
    print(f"\n{b}\n  {t}\n{b}")

def section(t):
    print(f"\n  {'─'*48}\n  {t}\n  {'─'*48}")

def ok(m):   print(f"  \033[92m✓\033[0m {m}")
def info(m): print(f"  {m}")
def warn(m): print(f"  \033[93m!\033[0m {m}")


# ---------------------------------------------------------------------------
# Pre-processamento (Utilizando MESES_SIZE importado)
# ---------------------------------------------------------------------------

def preprocess(path, target_size=MESES_SIZE):
    arr = np.array(Image.open(path).convert("L"))
    if (arr == 255).sum() < (arr == 0).sum():
        fg = (arr == 255).astype(np.uint8)
    else:
        fg = (arr == 0).astype(np.uint8)
    img = Image.fromarray(fg * 255)
    img = img.resize(target_size, Image.NEAREST)
    return (np.array(img) > 0).astype(np.uint8)


def zoning(fg, grid=GRID):
    H, W = fg.shape
    rows, cols = grid
    cell_h, cell_w = H // rows, W // cols
    feats = []
    for i in range(rows):
        for j in range(cols):
            block = fg[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # Média segura caso o bloco esteja vazio por arredondamento de redimensionamento
            feats.append(block.mean() if block.size > 0 else 0)
    return np.array(feats, dtype=np.float64)


def extract_handcrafted(path):
    # Usa o MESES_SIZE importado de handcrafted.py
    fg = preprocess(path, MESES_SIZE)
    return zoning(fg)


def extract_resnet(path, model, transform):
    import torch
    arr = np.array(Image.open(path).convert("L"))
    if (arr == 255).sum() < (arr == 0).sum():
        fg = (arr == 255).astype(np.uint8) * 255
    else:
        fg = (arr == 0).astype(np.uint8) * 255
    rgb = np.stack([fg, fg, fg], axis=-1)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().flatten()
    return feat


# ---------------------------------------------------------------------------
# Carregamento dos dados e KNN
# ---------------------------------------------------------------------------

def load_dataset(path):
    if not os.path.exists(path):
        return None, None
    data = np.loadtxt(path)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def knn_predict(X_train, y_train, x_query, k, top_n):
    """KNN puro em numpy — sem sklearn necessario."""
    dists = np.linalg.norm(X_train - x_query, axis=1)
    idx   = np.argsort(dists)[:k]
    vizinhos = y_train[idx]

    # contagem de votos por classe
    classes, votos = np.unique(vizinhos, return_counts=True)
    ordem = np.argsort(-votos)
    classes = classes[ordem]
    votos   = votos[ordem]

    top = [(LABEL_TO_MES.get(int(c), str(c)), int(v), v/k*100)
           for c, v in zip(classes[:top_n], votos[:top_n])]
    return top, dists[idx].mean()


def build_resnet_extractor():
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as T
    except ImportError:
        return None, None, None

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = "cpu"
    return model.to(device), transform, device


# ---------------------------------------------------------------------------
# Exibição do resultado
# ---------------------------------------------------------------------------

def print_result(titulo, top, dist_media, k):
    section(titulo)
    info(f"k = {k}  |  distância média dos {k} vizinhos: {dist_media:.4f}")
    print()
    for rank, (mes, votos, pct) in enumerate(top, 1):
        barra = "█" * int(pct / 5)
        destaque = "\033[1;96m" if rank == 1 else "\033[0m"
        reset    = "\033[0m"
        print(f"    {destaque}#{rank}  {mes:<12}  {votos:>2}/{k} votos  ({pct:5.1f}%)  {barra}{reset}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prevê o mês do ano a partir de uma imagem usando KNN."
    )
    parser.add_argument("imagem", help="Caminho para a imagem (.bmp / .png)")
    parser.add_argument("--k",   type=int, default=7,
                        help="Número de vizinhos do KNN (padrão: 7)")
    parser.add_argument("--top", type=int, default=3,
                        help="Quantos meses mostrar no ranking (padrão: 3)")
    args = parser.parse_args()

    if not os.path.exists(args.imagem):
        print(f"\n  ERRO: arquivo '{args.imagem}' não encontrado.")
        sys.exit(1)

    header("CLASSIFICADOR DE MESES  —  KNN")
    info(f"Imagem   : {args.imagem}")
    info(f"k        : {args.k}")
    info(f"Top      : {args.top} candidatos")
    info(f"Grid     : {GRID_ROWS}x{GRID_COLS} (Importado)")

    algum_resultado = False

    # ── HANDCRAFTED ──────────────────────────────────────────────────────────
    X_h, y_h = load_dataset(MESES_HANDCRAFTED)
    if X_h is not None:
        info("\nCarregando features handcrafted...")
        ok(f"{len(X_h)} amostras  |  {X_h.shape[1]} features")
        try:
            feat_h = extract_handcrafted(args.imagem)
            top_h, dist_h = knn_predict(X_h, y_h, feat_h, args.k, args.top)
            # Título dinâmico baseado no GRID importado
            print_result(f"RESULTADO — Handcrafted (Zoneamento {GRID_ROWS}×{GRID_COLS})", top_h, dist_h, args.k)
            algum_resultado = True
        except Exception as e:
            warn(f"Erro no handcrafted: {e}")
    else:
        warn(f"'{MESES_HANDCRAFTED}' não encontrado — rode extract_features.py primeiro.")

    # ── RESNET ────────────────────────────────────────────────────────────────
    X_r, y_r = load_dataset(MESES_RESNET)
    if X_r is not None:
        info("\nCarregando features ResNet18...")
        ok(f"{len(X_r)} amostras  |  {X_r.shape[1]} features")
        info("Carregando modelo ResNet18...")
        model, transform, device = build_resnet_extractor()
        if model is not None:
            ok(f"Modelo pronto  |  dispositivo: {device.upper()}")
            try:
                feat_r = extract_resnet(args.imagem, model, transform)
                top_r, dist_r = knn_predict(X_r, y_r, feat_r, args.k, args.top)
                print_result("RESULTADO — Non-Handcrafted (ResNet18)", top_r, dist_r, args.k)
                algum_resultado = True
            except Exception as e:
                warn(f"Erro no ResNet: {e}")
        else:
            warn("torch/torchvision não instalados — ResNet ignorada.")
    else:
        warn(f"'{MESES_RESNET}' não encontrado — rode extract_features.py primeiro.")

    if algum_resultado:
        print(f"\n  {'═'*52}")
        print(f"  Concluído.")
        print(f"  {'═'*52}\n")
    else:
        print("\n  Nenhum resultado — gere os arquivos de features primeiro.\n")


if __name__ == "__main__":
    main()