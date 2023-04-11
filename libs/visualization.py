import numpy as np
import libs.functions as _fn
from sklearn.decomposition import PCA
#from openTSNE import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(dataset=None, epoch=None, viz_dir=None):
    if viz_dir is None:
        raise Exception("Visualization dir not provided")
    if len(dataset) == 0:
        raise Exception("Empty dataset")
    if len(dataset) == 1:
        _fn.report("Warning: only 1 speaker provided for dataset")
    _fn.report("  Visualizing embeddings via TSNE")
    # fig = plt.figure(figsize=(15, 15))
    if epoch is not None:
        plt.title('Epoch ' + str(epoch))

    records = []
    colors = []
    cmap = plt.cm.get_cmap('hsv', len(dataset))
    for color, speakerData in enumerate(dataset):
        for _, spectrogram in enumerate(speakerData):
            records.append(spectrogram.flatten())
            colors.append(color)

    if len(records) > 0:
        records = np.array(records)
        pca = PCA(n_components=50)
        embeddingsPCA = pca.fit_transform(records)
        tsne = TSNE(n_components=2, perplexity=50)
        embeddingsTSNE = tsne.fit(embeddingsPCA)
        for spkIndex, _ in enumerate(dataset):
            filteredIndices = [
                index for index, color in enumerate(colors)
                if color == spkIndex
            ]
            recordsFiltered = embeddingsTSNE[filteredIndices]
            plt.scatter(recordsFiltered[:, 0],
                        recordsFiltered[:, 1],
                        s=1,
                        c=[cmap(spkIndex)] * len(recordsFiltered))
            for i, j in zip(recordsFiltered[:, 0], recordsFiltered[:, 1]):
                plt.annotate(str(spkIndex), xy=(i, j), fontsize='xx-small')
    plt.savefig(viz_dir + "epoch" + str(epoch) + ".png", dpi=300)  # display it
    plt.close()


def visualize3d(dataset=None, epoch=None, viz_dir=None):
    if viz_dir is None:
        raise Exception("Visualization dir not provided")
    if len(dataset) == 0:
        raise Exception("Empty dataset")
    if len(dataset) == 1:
        _fn.report("Warning: only 1 speaker provided for dataset")
    _fn.report("  Visualizing embeddings via TSNE")
    fig = plt.figure()
    ax = Axes3D(fig)
    if epoch is not None:
        plt.title('Epoch ' + str(epoch))

    records = []
    colors = []
    cmap = plt.cm.get_cmap('hsv', len(dataset))
    for color, speakerData in enumerate(dataset):
        for index, spectrogram in enumerate(speakerData):
            records.append(spectrogram.flatten())
            colors.append(color)

    if len(records) > 0:
        records = np.array(records)
        pca = PCA(n_components=50)
        embeddingsPCA = pca.fit_transform(records)
        tsne = TSNE(n_components=3, perplexity=50)
        embeddingsTSNE = tsne.fit(embeddingsPCA)
        for spkIndex, _ in enumerate(dataset):
            filteredIndices = [
                index for index, color in enumerate(colors)
                if color == spkIndex
            ]
            recordsFiltered = embeddingsTSNE[filteredIndices]
            ax.scatter(recordsFiltered[:, 0],
                       recordsFiltered[:, 1],
                       recordsFiltered[:, 2],
                       s=1,
                       c=[cmap(spkIndex)] * len(recordsFiltered))
    for i in range(0, 4):
        ax.view_init(azim=-90 + 90 * i)
        plt.savefig(viz_dir + "epoch" + str(epoch) + "_azim" + (str(i)) +
                    ".png")  # display it
    plt.close()


def distances_histogram(distances, filename):
    # fig = plt.figure()
    plt.hist(distances, bins=25, density=True, alpha=0.6, color='b')
    plt.savefig(filename)
    plt.close()
    return True


def graph(arr, filename, shape=None, title=None):
    if shape is None:
        shape = (10, 10)
    if arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    # fig = plt.figure(figsize=shape)
    if title is not None:
        plt.title(title)
    if arr.shape[0] == 28:
        colorMap = 'gray'
    else:
        colorMap = 'viridis'

    plt.imshow(arr, cmap=colorMap)
    plt.savefig(filename)
    plt.close()
    return True
