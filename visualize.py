"""
Visualizacao dos arquivos de features gerados.
Roda: python3 visualize.py
Gera 4 PNGs na pasta:
  - viz_handcrafted_grid.png  (reconstrucao 8x8 da grade de zoneamento)
  - viz_handcrafted_pca.png   (scatter 2D das 64 features handcrafted)
  - viz_resnet_pca.png        (scatter 2D das 512 features da ResNet)
  - viz_label_dist.png        (distribuicao de labels por arquivo)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # gera PNGs sem precisar de display
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.abspath(__file__))

DIGIT_LABELS = [str(i) for i in range(10)]
MES_LABELS = ["jan", "fev", "mar", "abr", "mai", "jun",
              "jul", "ago", "set", "out", "nov", "dez"]


def load(path):
    data = np.loadtxt(os.path.join(BASE, path))
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y


def head_preview(path, n=3, cols=8):
    print(f"\n--- {path} (primeiras {n} linhas, primeiras {cols} colunas + label) ---")
    X, y = load(path)
    for i in range(n):
        feats = " ".join(f"{v:7.4f}" for v in X[i, :cols])
        print(f"  [{feats} ... ]  label={y[i]}")
    print(f"  shape: {X.shape[0]} amostras x {X.shape[1]} features")


def plot_handcrafted_grid(file_in, label_names, out_png, title):
    """Reconstroi a grade 8x8 a partir das features e mostra a media por classe."""
    X, y = load(file_in)
    n_classes = len(label_names)
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(n_classes * 1.1, 4.2))
    axes = axes.flatten()
    for c in range(n_classes):
        mean_vec = X[y == c].mean(axis=0)
        grid = mean_vec.reshape(8, 8)
        axes[c].imshow(grid, cmap="gray_r", vmin=0, vmax=1)
        axes[c].set_title(f"{label_names[c]}", fontsize=10)
        axes[c].axis("off")
    for k in range(n_classes, len(axes)):
        axes[k].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {out_png}")


def plot_pca(file_in, label_names, out_png, title):
    X, y = load(file_in)
    Xp = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab20" if len(label_names) > 10 else "tab10")
    for c in range(len(label_names)):
        m = y == c
        ax.scatter(Xp[m, 0], Xp[m, 1], s=8, alpha=0.6, label=label_names[c], color=cmap(c))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2, loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {out_png}")


def plot_label_dist(files, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    axes = axes.flatten()
    for ax, (path, names) in zip(axes, files):
        _, y = load(path)
        counts = np.bincount(y, minlength=len(names))
        ax.bar(range(len(names)), counts, color="steelblue")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45 if len(names) > 10 else 0, fontsize=8)
        ax.set_title(path, fontsize=9)
        ax.set_ylabel("amostras")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  salvo: {out_png}")


def main():
    print("==================================================")
    print(" PREVIEW (texto)")
    print("==================================================")
    for f in ["ocr_handcrafted_treino.txt", "ocr_handcrafted_teste.txt", "meses_handcrafted.txt",
              "ocr_resnet_treino.txt", "ocr_resnet_teste.txt", "meses_resnet.txt"]:
        head_preview(f)

    print("\n==================================================")
    print(" PLOTS")
    print("==================================================")

    print("\n[1/4] Reconstrucao da grade 8x8 (handcrafted)")
    plot_handcrafted_grid("ocr_handcrafted_treino.txt", DIGIT_LABELS,
                          os.path.join(BASE, "viz_handcrafted_grid.png"),
                          "Media das features handcrafted por digito (8x8)")
    plot_handcrafted_grid("meses_handcrafted.txt", MES_LABELS,
                          os.path.join(BASE, "viz_handcrafted_grid_meses.png"),
                          "Media das features handcrafted por mes (8x8)")

    print("\n[2/4] PCA das features handcrafted")
    plot_pca("ocr_handcrafted_treino.txt", DIGIT_LABELS,
             os.path.join(BASE, "viz_handcrafted_pca.png"),
             "PCA 2D - OCR handcrafted (treino)")
    plot_pca("meses_handcrafted.txt", MES_LABELS,
             os.path.join(BASE, "viz_handcrafted_pca_meses.png"),
             "PCA 2D - Meses handcrafted")

    print("\n[3/4] PCA das features ResNet")
    plot_pca("ocr_resnet_treino.txt", DIGIT_LABELS,
             os.path.join(BASE, "viz_resnet_pca.png"),
             "PCA 2D - OCR ResNet18 (treino)")
    plot_pca("meses_resnet.txt", MES_LABELS,
             os.path.join(BASE, "viz_resnet_pca_meses.png"),
             "PCA 2D - Meses ResNet18")

    print("\n[4/4] Distribuicao de labels")
    files = [
        ("ocr_handcrafted_treino.txt", DIGIT_LABELS),
        ("ocr_handcrafted_teste.txt",  DIGIT_LABELS),
        ("meses_handcrafted.txt",      MES_LABELS),
        ("ocr_resnet_treino.txt",      DIGIT_LABELS),
        ("ocr_resnet_teste.txt",       DIGIT_LABELS),
        ("meses_resnet.txt",           MES_LABELS),
    ]
    plot_label_dist(files, os.path.join(BASE, "viz_label_dist.png"))

    print("\nPronto. Abra os arquivos viz_*.png na pasta.")


if __name__ == "__main__":
    main()
