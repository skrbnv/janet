import libs.data as _db
import cv2

SOURCE_DATASET = '/media/my3bikaht/EXT4/datasets/TIMIT2/datasets/glued-2s-mel80-plain-train.dt'
SOURCE_CACHE = '/media/my3bikaht/EXT4/datasets/TIMIT2/cache/glued-2s-mel80-plain'
TARGET_CACHE = '/media/my3bikaht/EXT4/datasets/TIMIT2/cache/glued-2s-mel80-224x224_plain'
TARGET_SHAPE = (224, 224)

D = _db.readDataset(SOURCE_DATASET)
D_len = len(D)
for i, (key, value) in enumerate(D.items()):
    spgm = _db.cacheRead(value['cacheId'], cache=SOURCE_CACHE)
    target = cv2.resize(spgm,
                        dsize=TARGET_SHAPE,
                        interpolation=cv2.INTER_LINEAR)
    _db.cacheWrite(value['cacheId'], target, cache=TARGET_CACHE)
    print("\r Resized %d out of %d spectrograms" % (i + 1, D_len), end='')

#_db.writeDataset(D, 'VCTK-train-80x200-upd.dt')
