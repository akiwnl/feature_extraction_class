"""
handcrafted.py
--------------
Módulo responsável pela extração de características HANDCRAFTED (feitas à mão).

Método utilizado: Zoneamento (Zoning)
  - A imagem é dividida em uma grade de GRID_ROWS x GRID_COLS células.
  - Para cada célula, calcula-se a fração de pixels de tinta (pixels ativos).
  - Isso gera um vetor de GRID_ROWS * GRID_COLS valores entre 0 e 1.

Por que zoneamento?
  - É simples, rápido e não precisa de GPU.
  - Captura a distribuição espacial dos pixels, o que é informativo
    para distinguir letras/dígitos com formatos diferentes.
  - É um dos métodos clássicos citados na atividade.
"""

import os
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constantes de configuração
# ---------------------------------------------------------------------------

# Tamanho para o qual as imagens OCR são redimensionadas antes da extração.
# Pequeno o suficiente para ser rápido, grande o suficiente para preservar
# a estrutura do caractere.
OCR_SIZE = (32, 32)

# Tamanho para imagens de meses. Mais largo pois palavras são horizontais.
MESES_SIZE = (128, 32)

# Dimensões da grade de zoneamento (linhas x colunas).
# Com 8x8 = 64 células → vetor de 64 features por imagem.
GRID_ROWS = 8
GRID_COLS = 8


# ---------------------------------------------------------------------------
# Utilitários de log
# ---------------------------------------------------------------------------

def section(title: str):
    """Imprime um cabeçalho de seção visível no terminal."""
    bar = "=" * 54
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def info(msg: str):
    """Mensagem informativa simples."""
    print(f"  {msg}")


def ok(msg: str):
    """Mensagem de sucesso destacada em verde."""
    print(f"  \033[92m✓\033[0m {msg}")


def warn(msg: str):
    """Mensagem de aviso destacada em amarelo."""
    print(f"  \033[93m!\033[0m {msg}")


# ---------------------------------------------------------------------------
# Pré-processamento de imagem
# ---------------------------------------------------------------------------

def preprocess(path: str, target_size: tuple) -> np.ndarray:
    """
    Carrega uma imagem e a converte para um mapa binário de pixels ativos.

    Passos:
      1. Abre e converte para escala de cinza (modo "L" do PIL).
      2. Detecta automaticamente se o fundo é branco ou preto,
         garantindo que tinta = 1 e fundo = 0 em qualquer caso.
      3. Redimensiona para target_size com interpolação NEAREST
         (sem suavização, preserva as bordas dos caracteres).

    Parâmetros:
      path        : caminho completo para o arquivo de imagem.
      target_size : tupla (largura, altura) desejada após o redimensionamento.

    Retorna:
      Array numpy 2D de uint8 — shape (altura, largura) — com valores 0 ou 1.
    """
    # Abre a imagem e converte para escala de cinza
    arr = np.array(Image.open(path).convert("L"))

    # Detecta a polaridade:
    #   - Se há mais pixels pretos (0) que brancos (255) → fundo preto, tinta branca
    #   - Caso contrário → fundo branco, tinta preta (situação mais comum)
    # Isso torna o código robusto para os dois tipos de imagens presentes nas bases.
    if (arr == 255).sum() < (arr == 0).sum():
        # Fundo preto → tinta são os pixels brancos (255)
        fg = (arr == 255).astype(np.uint8)
    else:
        # Fundo branco → tinta são os pixels pretos (0)
        fg = (arr == 0).astype(np.uint8)

    # Converte de volta para escala 0-255 para o PIL poder redimensionar
    img = Image.fromarray(fg * 255)
    # Redimensiona sem suavização para não criar valores intermediários (cinza)
    img = img.resize(target_size, Image.NEAREST)
    # Binariza novamente: qualquer valor > 0 vira 1
    return (np.array(img) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Extração de features: Zoneamento
# ---------------------------------------------------------------------------

def zoning(fg: np.ndarray) -> np.ndarray:
    """
    Aplica o zoneamento na imagem binária e retorna o vetor de features.

    Divide a imagem em GRID_ROWS × GRID_COLS células iguais e calcula
    a densidade de tinta (média dos pixels) em cada célula.

    Exemplo (grade 8x8 em imagem 32x32):
      - Cada célula tem 4×4 = 16 pixels.
      - Se 8 desses pixels forem tinta → feature da célula = 0.5.

    Por que isso funciona?
      - Caracteres diferentes têm distribuições de tinta distintas em
        diferentes regiões, tornando o vetor discriminativo para classificação.

    Parâmetros:
      fg : array 2D (H × W) com valores 0 ou 1 (saída de preprocess).

    Retorna:
      Array 1D de float64 com GRID_ROWS × GRID_COLS valores em [0.0, 1.0].
    """
    H, W = fg.shape

    # Tamanho de cada célula (divisão inteira para células uniformes)
    cell_h = H // GRID_ROWS
    cell_w = W // GRID_COLS

    # Vetor que receberá a densidade de tinta de cada célula
    feats = np.empty(GRID_ROWS * GRID_COLS, dtype=np.float64)

    k = 0  # índice linear: percorre as células da grade linha por linha
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            # Recorta a célula (i, j) da imagem binária
            block = fg[i * cell_h:(i + 1) * cell_h,
                       j * cell_w:(j + 1) * cell_w]
            # Média dos pixels = fração de tinta nessa região
            feats[k] = block.mean()
            k += 1

    return feats


# ---------------------------------------------------------------------------
# Pipeline de extração para uma única imagem
# ---------------------------------------------------------------------------

def extract_one(path: str, target_size: tuple) -> np.ndarray:
    """
    Executa o pipeline completo de extração handcrafted para uma imagem.

    Encapsula preprocess + zoning em uma única chamada conveniente,
    usada tanto pelo write_features quanto pelo predict_mes.py.

    Parâmetros:
      path        : caminho para a imagem.
      target_size : tamanho de redimensionamento (OCR_SIZE ou MESES_SIZE).

    Retorna:
      Vetor 1D de features com GRID_ROWS × GRID_COLS elementos.
    """
    fg = preprocess(path, target_size)  # binariza e redimensiona
    return zoning(fg)                   # calcula densidades por zona


# ---------------------------------------------------------------------------
# Escrita do arquivo de saída
# ---------------------------------------------------------------------------

def write_features(items: list, target_size: tuple, out_path: str):
    """
    Processa todas as imagens da lista e salva as features em um arquivo .txt.

    Formato de saída — uma linha por imagem:
      feat_0 feat_1 ... feat_63 label
      0.062500 0.000000 ... 0.312500 3

    O label fica sempre na última coluna, seguindo o padrão da atividade.

    Parâmetros:
      items       : lista de tuplas (caminho_imagem, label_inteiro).
      target_size : tamanho de redimensionamento (OCR_SIZE ou MESES_SIZE).
      out_path    : caminho completo do arquivo .txt de saída.
    """
    t0 = time.time()

    # Número de features = total de células da grade
    n_feats = GRID_ROWS * GRID_COLS

    # Pré-aloca a matriz completa em memória para salvar tudo de uma vez
    # (mais eficiente do que escrever linha a linha em disco)
    rows = np.empty((len(items), n_feats + 1), dtype=np.float64)

    # Barra de progresso com velocidade (img/s) e tempo estimado restante
    pbar = tqdm(
        enumerate(items),
        total=len(items),
        desc="    Imagens",
        unit="img",
        ncols=70,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    erros = 0  # conta imagens que falharam (corrompidas, formato errado etc.)
    for i, (path, label) in pbar:
        try:
            rows[i, :-1] = extract_one(path, target_size)  # features nas colunas 0..N-1
            rows[i, -1]  = label                            # label na última coluna
        except Exception as e:
            warn(f"Erro em {os.path.basename(path)}: {e}")
            rows[i] = 0   # linha zerada para não corromper o arquivo
            erros += 1

    # %.6f para features (6 casas decimais), %d para o label inteiro
    fmt = ["%.6f"] * n_feats + ["%d"]
    np.savetxt(out_path, rows, fmt=fmt)

    elapsed = time.time() - t0
    ok(
        f"Salvo: {os.path.basename(out_path)}  "
        f"({len(items)} amostras | {n_feats} features | {elapsed:.1f}s)"
        + (f"  [{erros} erro(s)]" if erros else "")
    )