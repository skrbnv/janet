from torch.utils import data as D
import numpy as np


class SpeakerDataset(D.Dataset):
    def __init__(self,
                 inputs,
                 speakerIDs,
                 sampleIDs,
                 positionIDs,
                 embeddings=None,
                 batch_size=64):
        'Initialization'
        if inputs.shape[0] == len(speakerIDs) == len(sampleIDs) == len(
                positionIDs):
            pass
        else:
            print("Size mismatch while creating dataset", inputs.shape,
                  len(speakerIDs), len(sampleIDs), len(positionIDs))
            raise Exception("Inputs and rest arrays must have same size")
        modulo = -1 * (inputs.shape[0] % batch_size)
        self.inputs = inputs[:modulo, ]
        self.speakerIDs = speakerIDs[:modulo]
        self.sampleIDs = sampleIDs[:modulo]
        self.positionIDs = positionIDs[:modulo]
        self.embeddings = np.zeros([inputs.shape[0], 128])
        # 128 is the default size of embeddding vector
        print("-- Dataset created using", self.inputs.shape[0],
              "spectrograms out of", inputs.shape[0], "total --")

    def __len__(self):
        'Denotes the total number of sampleIDs'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        input = self.inputs[index]
        speakerID = self.speakerIDs[index]
        sampleID = self.sampleIDs[index]
        positionID = self.positionIDs[index]
        embedding = self.embeddings[index]
        return input, speakerID, sampleID, positionID, embedding

    def report(self, index):
        'Provides info on element by index'
        speakerID = self.speakerIDs[index]
        sampleID = self.sampleIDs[index]
        positionID = self.positionIDs[index]
        print("Dataset element: ", speakerID, sampleID, positionID)

    def set_embedding(self, embedding, index):
        'Updates embedding by index'
        self.embeddings[index] = embedding

    def get_embedding(self, index):
        'Returns embedding by index'
        return self.embeddings[index]

    def get_unique_speakers(self):
        'Returns list of unique speakers'
        speakers = []
        for i in range(len(self.speakerIDs)):
            if any(self.speakerIDs[i] == speaker for speaker in speakers):
                pass
            else:
                speakers.append(self.speakerIDs[i])
        return speakers

    def get_items_by_speaker(self, speakerID):
        'Returns elements by speaker ID'
        ixs = [i for i, x in enumerate(self.speakerIDs) if x == speakerID]
        return [self.inputs[i]
                for i in ixs], [self.speakerIDs[i] for i in ixs
                                ], [self.sampleIDs[i] for i in ixs
                                    ], [self.positionIDs[i] for i in ixs
                                        ], [self.embeddings[i] for i in ixs]

    def get_positive_samples(self, speakerID, sampleID, positionID):
        'Returns positive samples for specific item (spectrogram)'
        ixs = [i for i, x in enumerate(self.speakerIDs) if x == speakerID]
        ixs.pop([
            j for j, y in enumerate(ixs) if self.sampleIDs[y] == sampleID
            and self.positionIDs[y] == positionID
        ][0])
        return [self.inputs[i]
                for i in ixs], [self.speakerIDs[i] for i in ixs
                                ], [self.sampleIDs[i] for i in ixs
                                    ], [self.positionIDs[i] for i in ixs
                                        ], [self.embeddings[i] for i in ixs]


class TripletDataset(D.Dataset):
    def __init__(self, anchors, positives, negatives):
        if (len(anchors) == len(positives) == len(negatives)):
            pass
        else:
            print("Size mismatch while creating dataset", len(anchors),
                  len(positives), len(negatives))
            raise Exception("Inputs and rest arrays must have same size")
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = self.anchors[index]
        positive = self.positives[index]
        negative = self.negatives[index]
        return anchor, positive, negative


class TripletDualDataset(D.Dataset):
    def __init__(self, anchors, positives, negatives):
        if (len(anchors) == len(positives) == len(negatives)):
            pass
        else:
            print("Size mismatch while creating dataset", len(anchors),
                  len(positives), len(negatives))
            raise Exception("Inputs and rest arrays must have same size")
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = self.anchors[index]
        positive = self.positives[index]
        negative = self.negatives[index]
        return anchor, positive, negative


class CentroidDataset(D.Dataset):
    def __init__(self, siblings):
        self.siblings = siblings

    def __len__(self):
        return len(self.siblings)

    def __getitem__(self, index):
        return self.siblings[index]


class BasicDataset(D.Dataset):
    def __init__(self, D):
        self.D = D
        speakers = self.D.get_unique_speakers()
        length = len(speakers)
        for i in range(D.length()):
            idx = speakers.index(self.D.data[i]['speaker'])
            add = np.zeros(length)
            add[idx] = 1
            D.data[i]['class'] = add

    def __len__(self):
        return self.D.length()

    def __getitem__(self, index):
        return (self.D.cache_read(self.D.data[index]['cacheId']),
                self.D.data[index]['class'])
