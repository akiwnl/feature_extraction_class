"""
non_handcrafted.py
------------------
Módulo responsável pela extração de características NON-HANDCRAFTED,
ou seja, features aprendidas automaticamente por uma rede neural.

Método utilizado: ResNet18 pré-treinada (ImageNet) como extrator
  - A camada classificadora final (fc) é removida.
  - A rede é usada somente para "olhar" a imagem e devolver um vetor
    de 512 números que resumem o seu conteúdo visual.
  - Esse vetor é a representação aprendida pela rede durante o treino
    no ImageNet, e se mostra útil mesmo para domínios diferentes.

Por que ResNet18?
  - É leve (< 50 MB), roda razoavelmente em CPU.
  - Produz features de qualidade superior ao zoneamento manual.
  - Está disponível no torchvision sem download manual de pesos.
"""

import os
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constantes de configuração
# ---------------------------------------------------------------------------

# Tamanho de entrada da ResNet18. A rede espera imagens 224×224.
RESNET_INPUT_SIZE = (224, 224)

# Número de features produzidas pela ResNet18 sem a camada final.
RESNET_FEAT_DIM = 512

# Número de imagens processadas por vez na GPU/CPU.
# Valores maiores são mais rápidos, mas consomem mais memória.
# Reduzir para 16 ou 32 se ocorrer erro de memória.
BATCH_SIZE = 64

# Normalização padrão do ImageNet (média e desvio por canal RGB).
# Obrigatória pois os pesos foram treinados com essa normalização.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Pipeline de transformação aplicado a cada imagem antes de entrar na rede.
# Usando NEAREST para não criar valores intermediários em imagens binárias.
RESNET_TFM = T.Compose([
    # Redimensiona para 224×224 (tamanho esperado pela ResNet)
    T.Resize(RESNET_INPUT_SIZE, interpolation=T.InterpolationMode.NEAREST),
    # Converte PIL Image para tensor PyTorch (valores em [0.0, 1.0])
    T.ToTensor(),
    # Normaliza com média e desvio do ImageNet — obrigatório para pesos pré-treinados
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Construção do modelo extrator
# ---------------------------------------------------------------------------

def build_resnet() -> tuple:
    """
    Carrega a ResNet18 pré-treinada e adapta para extração de features.

    Modificação principal:
      - A última camada (model.fc), que classifica em 1000 classes ImageNet,
        é substituída por nn.Identity(), que simplesmente repassa a entrada.
      - Isso faz a rede retornar o vetor de 512 features em vez de 1000 logits.

    O modelo é colocado em modo eval() para:
      - Desativar Dropout (não queremos variação aleatória na extração).
      - Usar estatísticas fixas nas BatchNorm layers.

    Retorna:
      Tupla (model, device) onde:
        model  : ResNet18 pronta para extração, já no dispositivo correto.
        device : string "cuda" ou "cpu".
    """
    print(f"  Carregando ResNet18 pré-treinada (ImageNet)...")

    # Carrega os pesos pré-treinados do ImageNet1K
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Remove o classificador final → rede vira um extrator de 512 features
    # nn.Identity() retorna a entrada sem modificação
    model.fc = nn.Identity()

    # Modo de avaliação: desativa dropout e usa BatchNorm com estatísticas fixas
    model.eval()

    # Usa GPU se disponível, caso contrário usa CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"  \033[92m✓\033[0m Modelo carregado  |  dispositivo: {device.upper()}")
    return model, device


# ---------------------------------------------------------------------------
# Pipeline de pré-processamento de uma imagem para a ResNet
# ---------------------------------------------------------------------------

def prepare_image(path: str) -> torch.Tensor:
    """
    Carrega uma imagem, corrige a polaridade (tinta/fundo) e aplica
    as transformações necessárias para entrada na ResNet18.

    A ResNet espera imagens RGB, mas nossas bases são binárias (preto e branco).
    Solução: triplicamos o canal de cinza para simular RGB (R=G=B=tinta).

    Parâmetros:
      path : caminho para a imagem (.bmp ou similar).

    Retorna:
      Tensor de shape (3, 224, 224) pronto para ser empilhado em batch.
    """
    arr = np.array(Image.open(path).convert("L"))

    # Detecta polaridade: garante que tinta = 255 (branco) e fundo = 0
    if (arr == 255).sum() < (arr == 0).sum():
        # Fundo preto, tinta branca → mantém os brancos
        fg = (arr == 255).astype(np.uint8) * 255
    else:
        # Fundo branco, tinta preta → inverte para tinta ficar em 255
        fg = (arr == 0).astype(np.uint8) * 255

    # Cria imagem RGB triplicando o canal (R=G=B) para compatibilidade com ResNet
    rgb = np.stack([fg, fg, fg], axis=-1)   # shape: (H, W, 3)

    # Aplica redimensionamento, ToTensor e normalização ImageNet
    return RESNET_TFM(Image.fromarray(rgb))  # shape: (3, 224, 224)


# ---------------------------------------------------------------------------
# Extração de features em batch
# ---------------------------------------------------------------------------

def extract_features(model: nn.Module, device: str, paths: list) -> np.ndarray:
    """
    Extrai vetores de 512 features para uma lista de imagens usando batches.

    Por que usar batches?
      - Processar imagens uma a uma é lento (overhead de transferência GPU).
      - Em batch, a GPU/CPU processa várias imagens em paralelo.

    torch.no_grad() desativa o cálculo de gradientes, economizando memória
    e acelerando a inferência (não precisamos de gradientes para extração).

    Parâmetros:
      model  : ResNet18 sem a camada final (saída de build_resnet).
      device : "cuda" ou "cpu".
      paths  : lista de caminhos de imagens.

    Retorna:
      Array numpy de shape (N, 512) com as features de todas as imagens.
    """
    # Pré-aloca o array de saída completo para evitar realocações em memória
    feats_all = np.empty((len(paths), RESNET_FEAT_DIM), dtype=np.float32)

    # Número total de batches (para a barra de progresso)
    n_batches = (len(paths) + BATCH_SIZE - 1) // BATCH_SIZE

    with torch.no_grad():  # desativa gradientes → mais rápido e menos memória
        pbar = tqdm(
            range(0, len(paths), BATCH_SIZE),
            total=n_batches,
            desc="    Batches",
            unit="batch",
            ncols=70,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for start in pbar:
            # Seleciona o grupo de imagens deste batch
            chunk = paths[start:start + BATCH_SIZE]

            # Prepara cada imagem e empilha em um tensor de batch
            tensors = []
            for p in chunk:
                tensors.append(prepare_image(p))
            batch = torch.stack(tensors, dim=0).to(device)  # (B, 3, 224, 224)

            # Inferência: batch entra na ResNet e sai um tensor (B, 512)
            out = model(batch).cpu().numpy()
            feats_all[start:start + len(chunk)] = out

            # Atualiza a barra com o número de imagens processadas
            imgs_done = min(start + BATCH_SIZE, len(paths))
            pbar.set_postfix({"imgs": f"{imgs_done}/{len(paths)}"}, refresh=True)

    return feats_all


# ---------------------------------------------------------------------------
# Extração para uma única imagem (usado pelo predict_mes.py)
# ---------------------------------------------------------------------------

def extract_one(path: str, model: nn.Module, device: str) -> np.ndarray:
    """
    Extrai o vetor de 512 features de uma única imagem.

    Usado pelo predict_mes.py para processar a imagem enviada pelo usuário
    sem precisar de uma lista completa de caminhos.

    Parâmetros:
      path   : caminho para a imagem.
      model  : ResNet18 sem camada final.
      device : "cuda" ou "cpu".

    Retorna:
      Array 1D de float32 com 512 valores.
    """
    tensor = prepare_image(path).unsqueeze(0).to(device)  # adiciona dimensão de batch
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().flatten()       # remove dimensões extras
    return feat


# ---------------------------------------------------------------------------
# Escrita do arquivo de saída
# ---------------------------------------------------------------------------

def write_features(items: list, model: nn.Module, device: str, out_path: str):
    """
    Extrai features de todas as imagens da lista e salva em um arquivo .txt.

    Formato de saída — uma linha por imagem:
      feat_0 feat_1 ... feat_511 label
      0.123456 0.000000 ... 0.987654 5

    O label fica sempre na última coluna, seguindo o padrão da atividade.

    Parâmetros:
      items    : lista de tuplas (caminho_imagem, label_inteiro).
      model    : ResNet18 sem camada final.
      device   : "cuda" ou "cpu".
      out_path : caminho completo do arquivo .txt de saída.
    """
    t0 = time.time()

    # Separa caminhos e labels para facilitar o processamento em batch
    paths  = [p for p, _ in items]
    labels = np.array([l for _, l in items], dtype=np.int64).reshape(-1, 1)

    # Extrai as features para todas as imagens de uma vez (em batches internamente)
    feats = extract_features(model, device, paths)

    # Concatena features (N, 512) com labels (N, 1) → matriz (N, 513)
    rows = np.hstack([feats.astype(np.float64), labels])

    # Salva: 6 casas decimais para features, inteiro para label
    fmt = ["%.6f"] * RESNET_FEAT_DIM + ["%d"]
    np.savetxt(out_path, rows, fmt=fmt)

    elapsed = time.time() - t0
    print(
        f"  \033[92m✓\033[0m Salvo: {os.path.basename(out_path)}  "
        f"({len(items)} amostras | {RESNET_FEAT_DIM} features | {elapsed:.1f}s)"
    )