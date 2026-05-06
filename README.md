# Trabalho – Extração de Características

**Disciplina:** Inteligência Computacional
**Professor:** Prof. Dr. Diego Bertolini
**Instituição:** UTFPR – Campus Campo Mourão – DACOM
**Entrega:** 04/05/2026

**Aluno(a):** _<preencher: nome completo>_
**RA:** _<preencher>_
**Dupla (se houver):** _<preencher ou remover>_

---

## 1. Objetivo

Extrair dois conjuntos de características das bases **OCR (dígitos manuscritos)** e **Meses do Ano**, gerando vetores que serão usados em etapas posteriores de classificação. Para cada base, são extraídos:

- **Características handcrafted** – projetadas manualmente.
- **Características non-handcrafted** – obtidas a partir de uma rede neural pré-treinada.

## 2. Arquivos entregues

Seis arquivos `.txt` no formato `Atributo1 Atributo2 ... AtributoN Label`:

| Arquivo | Base | Tipo | Amostras | Atributos |
|---|---|---|---|---|
| `ocr_handcrafted_treino.txt` | OCR | Handcrafted | 1000 | 64 |
| `ocr_handcrafted_teste.txt` | OCR | Handcrafted | 1000 | 64 |
| `ocr_resnet_treino.txt` | OCR | Non-handcrafted | 1000 | 512 |
| `ocr_resnet_teste.txt` | OCR | Non-handcrafted | 1000 | 512 |
| `meses_handcrafted.txt` | Meses | Handcrafted | 6000 | 64 |
| `meses_resnet.txt` | Meses | Non-handcrafted | 6000 | 512 |

Acompanha também:

- `extract_features.py` – script que gera os 6 arquivos.
- `visualize.py` – script que gera os PNGs de visualização.
- `viz_*.png` – figuras (médias por classe, PCA 2D e distribuição de labels).

## 3. Pré-processamento

Aplicado de forma uniforme nas duas bases:

1. Leitura como imagem em escala de cinza.
2. Detecção automática do *foreground*: a cor (0 ou 255) com menor quantidade de pixels é considerada tinta. Isso resolve a inversão de polaridade observada (OCR tem fundo preto e tinta branca; Meses tem fundo branco e tinta preta).
3. Conversão para máscara binária (1 = tinta, 0 = fundo).
4. Redimensionamento com interpolação **NEAREST** (mantém os valores binários):
   - **OCR:** 32 × 32
   - **Meses:** 128 × 32 (preserva razão de aspecto das palavras)

## 4. Extração handcrafted – Zoneamento 8 × 8

A imagem pré-processada é dividida em uma grade de 8 × 8 = **64 zonas**. Cada zona contribui com um valor: a **fração de pixels de tinta** dentro daquela célula (entre 0 e 1).

Vantagens: simples de implementar, robusto à variação de tamanho original, captura distribuição espacial de tinta. É a feature recomendada explicitamente no enunciado (zoneamento, contagem de pixels).

## 5. Extração non-handcrafted – ResNet18

Utiliza a arquitetura **ResNet18 pré-treinada na ImageNet** (`torchvision.models`). A camada de classificação final (`fc`) é substituída por `nn.Identity()`, fazendo com que a saída seja o vetor da camada de *global average pooling* — **512 dimensões**.

Procedimento por imagem:

1. Imagem binária convertida em três canais idênticos (RGB).
2. Redimensionamento para 224 × 224.
3. Normalização com média e desvio-padrão da ImageNet.
4. *Forward* na rede em modo `eval()` (sem treino), obtendo o vetor de 512 floats.

A ResNet é uma das arquiteturas explicitamente listadas no enunciado (*"VGG, ResNet, ViT, outros"*).

## 6. Labels

- **OCR:** o label é o nome do diretório (`0`–`9`).
- **Meses:** o label é derivado do prefixo do nome do arquivo, segundo o mapeamento:

  | Prefixo | Mês | Label |
  |---|---|---|
  | `j` | janeiro | 0 |
  | `f` | fevereiro | 1 |
  | `m` | março | 2 |
  | `a` | abril | 3 |
  | `md` | maio | 4 |
  | `jd` | junho | 5 |
  | `jt` | julho | 6 |
  | `ad` | agosto | 7 |
  | `s` | setembro | 8 |
  | `o` | outubro | 9 |
  | `n` | novembro | 10 |
  | `d` | dezembro | 11 |

  O parser testa primeiro os prefixos de duas letras (`md`, `jd`, `jt`, `ad`) para evitar confundir, por exemplo, `md` com `m`.

## 7. Como reproduzir

```bash
# instala dependências
pip install numpy pillow torch torchvision matplotlib scikit-learn

# gera os 6 arquivos .txt
python3 extract_features.py

# gera as visualizações (.png)
python3 visualize.py
```

Tempo aproximado em CPU: ~2 min handcrafted, ~15 min ResNet (8000 imagens).

## 8. O que pode ser feito para melhorar o desempenho

Pontos para explorar nas próximas etapas (classificação):

1. **Aumentar a granularidade do handcrafted** – grade 16 × 16 (256 features) ou concatenar zoneamento com histogramas verticais/horizontais e momentos de Hu.
2. **Centralização do dígito/palavra** – remover bordas em branco antes de redimensionar (*tight crop*) reduz variação espacial e melhora o zoneamento.
3. **Normalização das features** – aplicar *z-score* ou *min-max* antes do classificador, principalmente para as features da ResNet, cujas magnitudes variam bastante.
4. **Trocar a arquitetura non-handcrafted** – ViT (`google/vit-base-patch16-224`) ou ResNet50 produzem representações mais ricas que ResNet18.
5. **Fine-tuning** – em vez de usar a ResNet apenas como extrator congelado, treinar a última(s) camada(s) na própria base produziria features bem mais discriminativas.
6. **Data augmentation** – pequenas rotações, translações e ruído durante o fine-tuning aumentam a robustez.
7. **Redução de dimensionalidade** – aplicar PCA nas 512 features da ResNet para acelerar o classificador e atenuar o *curse of dimensionality*.
8. **Ensemble handcrafted + non-handcrafted** – concatenar os dois vetores (64 + 512 = 576 features) tende a superar cada um isoladamente.
