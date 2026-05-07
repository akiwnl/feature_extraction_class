"""
main.py
-------
Ponto de entrada principal do trabalho de Extração de Características.
Disciplina: Inteligência Computacional — UTFPR Campo Mourão

Coordena a execução dos dois módulos de extração:
  1. handcrafted.py    → Zoneamento 8×8 (64 features por imagem)
  2. non_handcrafted.py → ResNet18 pré-treinada (512 features por imagem)

Bases de dados esperadas na mesma pasta que este arquivo:
  treino/          → subpastas 0..9 com imagens OCR de treinamento
  teste/           → subpastas 0..9 com imagens OCR de teste
  Base de Dados - Meses do Ano/  → todas as imagens de meses juntas

Arquivos gerados (6 no total):
  ocr_handcrafted_treino.txt
  ocr_handcrafted_teste.txt
  meses_handcrafted.txt
  ocr_resnet_treino.txt
  ocr_resnet_teste.txt
  meses_resnet.txt
"""

import os
import time
from collections import Counter

# Importa os dois módulos de extração criados separadamente.
# Cada módulo cuida de sua própria lógica e pode ser usado de forma independente.
import handcrafted    as hc   # extração manual: zoneamento
import non_handcrafted as nhc  # extração automática: ResNet18

# ---------------------------------------------------------------------------
# Caminhos das bases de dados
# ---------------------------------------------------------------------------

# BASE aponta para o diretório onde este script está localizado.
# Todos os outros caminhos são relativos a ele, o que garante que o código
# funciona independente de onde o usuário execute o script.
BASE = os.path.dirname(os.path.abspath(__file__))

# Pasta com imagens OCR separadas por classe (subpastas 0..9) — para treino
OCR_TREINO_DIR = os.path.join(BASE, "treino")

# Pasta com imagens OCR separadas por classe (subpastas 0..9) — para teste
OCR_TESTE_DIR  = os.path.join(BASE, "teste")

# Pasta com TODAS as imagens de meses juntas (sem subpastas).
# O label é inferido do prefixo do nome do arquivo (ex: "jd" → Junho).
MESES_DIR = os.path.join(BASE, "Base de Dados - Meses do Ano")

# ---------------------------------------------------------------------------
# Mapeamento de prefixos de nome de arquivo → label numérico (meses)
# ---------------------------------------------------------------------------

# Cada prefixo corresponde a um mês. Os prefixos mais longos são verificados
# primeiro (ver list_meses) para evitar ambiguidade entre "j" e "jd"/"jt".
MES_LABEL = {
    "j":  0,   # Janeiro
    "f":  1,   # Fevereiro
    "m":  2,   # Março
    "a":  3,   # Abril
    "md": 4,   # Maio  (prefixo de 2 letras para evitar conflito com "m" de Março)
    "jd": 5,   # Junho (prefixo de 2 letras para evitar conflito com "j" de Janeiro)
    "jt": 6,   # Julho (prefixo de 2 letras para evitar conflito com "j" de Janeiro)
    "ad": 7,   # Agosto (prefixo de 2 letras para evitar conflito com "a" de Abril)
    "s":  8,   # Setembro
    "o":  9,   # Outubro
    "n":  10,  # Novembro
    "d":  11,  # Dezembro
}

# Ordena prefixos do mais longo ao mais curto para verificar primeiro os
# mais específicos ("jd", "jt") antes dos genéricos ("j").
MES_PREFIXES_SORTED = sorted(MES_LABEL.keys(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Funções de listagem das bases de dados
# ---------------------------------------------------------------------------

def list_ocr(folder: str) -> list:
    """
    Percorre a pasta OCR e retorna lista de (caminho, label) para cada imagem.

    Estrutura esperada:
      folder/
        0/   img1.bmp  img2.bmp ...   ← label = 0
        1/   img1.bmp  img2.bmp ...   ← label = 1
        ...
        9/   img1.bmp  img2.bmp ...   ← label = 9

    O label é o próprio nome da subpasta convertido para inteiro.

    Parâmetros:
      folder : caminho da pasta raiz (OCR_TREINO_DIR ou OCR_TESTE_DIR).

    Retorna:
      Lista de tuplas (caminho_absoluto_imagem, label_inteiro).
    """
    items = []
    # Itera sobre as subpastas em ordem alfabética (0, 1, 2, ..., 9)
    for cls in sorted(os.listdir(folder)):
        cls_dir = os.path.join(folder, cls)

        # Ignora arquivos soltos na pasta raiz (apenas subpastas importam)
        if not os.path.isdir(cls_dir):
            continue

        # Filtra apenas arquivos .bmp e os ordena para reprodutibilidade
        files = [f for f in sorted(os.listdir(cls_dir)) if f.lower().endswith(".bmp")]
        for f in files:
            items.append((os.path.join(cls_dir, f), int(cls)))

    return items


def label_from_meses_name(filename: str) -> int:
    """
    Determina o label (inteiro) de uma imagem de mês a partir do nome do arquivo.

    Estratégia: verifica prefixos do mais longo ao mais curto para evitar
    que "jd" seja confundido com "j".

    Exemplo:
      "jd168.bmp" → prefixo "jd" → label 5 (Junho)
      "a100.bmp"  → prefixo "a"  → label 3 (Abril)

    Parâmetros:
      filename : nome do arquivo (pode incluir o caminho completo).

    Retorna:
      Inteiro com o label do mês.

    Lança:
      ValueError se nenhum prefixo conhecido for encontrado no nome.
    """
    name = os.path.basename(filename)  # remove o caminho, usa só o nome do arquivo

    for pref in MES_PREFIXES_SORTED:
        # Verifica se o nome começa com o prefixo E que o próximo caractere é dígito
        # (evita falsos positivos como "marco_info.bmp" sendo classificado como "m")
        if name.startswith(pref) and name[len(pref):len(pref) + 1].isdigit():
            return MES_LABEL[pref]

    raise ValueError(f"Prefixo não reconhecido: {name}")


def list_meses(folder: str) -> list:
    """
    Lista todos os arquivos .bmp da pasta de meses e retorna (caminho, label).

    Como a pasta não tem subpastas, o label é extraído do nome do arquivo
    pela função label_from_meses_name.

    Parâmetros:
      folder : caminho da pasta com todas as imagens de meses.

    Retorna:
      Lista de tuplas (caminho_absoluto_imagem, label_inteiro).
    """
    items = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".bmp"):
            items.append((os.path.join(folder, f), label_from_meses_name(f)))
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """
    Função principal: lista as bases, extrai e salva as 6 combinações de features.

    Fluxo:
      1. Lista as imagens das 3 bases (OCR treino, OCR teste, Meses).
      2. Exibe um resumo com número de imagens e distribuição de classes.
      3. Executa a extração handcrafted (zoneamento) para as 3 bases.
      4. Carrega a ResNet18 e executa a extração non-handcrafted para as 3 bases.
      5. Exibe um resumo final com os arquivos gerados e seus tamanhos.
    """
    t_total = time.time()  # marca o tempo inicial para calcular o tempo total ao final

    # ── 1. Listagem das bases ────────────────────────────────────────────────
    hc.section("LISTANDO BASES DE DADOS")

    ocr_treino = list_ocr(OCR_TREINO_DIR)
    ocr_teste  = list_ocr(OCR_TESTE_DIR)
    meses      = list_meses(MESES_DIR)

    # Conta quantas imagens existem por classe em cada base
    # Counter retorna um dicionário {classe: quantidade}
    dist_treino = Counter(l for _, l in ocr_treino)
    dist_teste  = Counter(l for _, l in ocr_teste)
    dist_meses  = Counter(l for _, l in meses)

    # Exibe o resumo para que o usuário possa verificar se as bases foram
    # lidas corretamente antes de iniciar o processamento demorado
    hc.info(f"OCR treino : {len(ocr_treino):>5} imagens  |  classes: { {k: dist_treino[k] for k in sorted(dist_treino)} }")
    hc.info(f"OCR teste  : {len(ocr_teste):>5} imagens  |  classes: { {k: dist_teste[k]  for k in sorted(dist_teste)}  }")
    hc.info(f"Meses      : {len(meses):>5} imagens  |  {len(dist_meses)} classes | labels: {sorted(dist_meses.keys())}")

    # ── 2. Extração Handcrafted ──────────────────────────────────────────────
    hc.section("1/2  HANDCRAFTED  —  Zoneamento 8×8  (64 features)")

    hc.info("Processando OCR treino...")
    hc.write_features(
        ocr_treino,
        hc.OCR_SIZE,
        os.path.join(BASE, "ocr_handcrafted_treino.txt")
    )

    hc.info("\nProcessando OCR teste...")
    hc.write_features(
        ocr_teste,
        hc.OCR_SIZE,
        os.path.join(BASE, "ocr_handcrafted_teste.txt")
    )

    hc.info("\nProcessando Meses...")
    hc.write_features(
        meses,
        hc.MESES_SIZE,
        os.path.join(BASE, "meses_handcrafted.txt")
    )

    # ── 3. Extração Non-Handcrafted ──────────────────────────────────────────
    hc.section("2/2  NON-HANDCRAFTED  —  ResNet18  (512 features)")

    # Carrega o modelo uma única vez e reutiliza nas 3 bases
    # (evita carregar os ~45 MB de pesos três vezes)
    model, device = nhc.build_resnet()

    hc.info("\nProcessando OCR treino...")
    nhc.write_features(
        ocr_treino,
        model,
        device,
        os.path.join(BASE, "ocr_resnet_treino.txt")
    )

    hc.info("\nProcessando OCR teste...")
    nhc.write_features(
        ocr_teste,
        model,
        device,
        os.path.join(BASE, "ocr_resnet_teste.txt")
    )

    hc.info("\nProcessando Meses...")
    nhc.write_features(
        meses,
        model,
        device,
        os.path.join(BASE, "meses_resnet.txt")
    )

    # ── 4. Resumo final ──────────────────────────────────────────────────────
    hc.section("CONCLUÍDO")

    elapsed_total = time.time() - t_total
    hc.info(f"Tempo total: {elapsed_total:.1f}s")
    hc.info("Arquivos gerados:")

    # Lista os 6 arquivos com seus tamanhos em KB para confirmar que foram criados
    arquivos = [
        "ocr_handcrafted_treino.txt",
        "ocr_handcrafted_teste.txt",
        "meses_handcrafted.txt",
        "ocr_resnet_treino.txt",
        "ocr_resnet_teste.txt",
        "meses_resnet.txt",
    ]
    for nome in arquivos:
        path = os.path.join(BASE, nome)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # converte bytes → KB
            hc.ok(f"{nome}  ({size:.0f} KB)")
        else:
            hc.warn(f"{nome}  NÃO GERADO")

    print()


# Garante que main() só é chamado quando o script é executado diretamente,
# não quando é importado por outro módulo (ex: pelo predict_mes.py).
if __name__ == "__main__":
    main()