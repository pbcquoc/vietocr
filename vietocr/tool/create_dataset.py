import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def createDataset(outputPath, root_dir, annotation_path):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    annotation_path = os.path.join(root_dir, annotation_path)
    with open(annotation_path, 'r') as ann_file:
        lines = ann_file.readlines()
        annotations = [l.strip().split('\t') for l in lines]

    nSamples = len(annotations)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 0
    for i in range(nSamples):
        imageFile, label = annotations[i]
        imagePath = os.path.join(root_dir, imageFile)

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
#        if checkValid:
#            if not checkImageIsValid(imageBin):
#                print('%s is not a valid image' % imagePath)
#                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        pathKey = 'path-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[pathKey] = imageFile.encode()
        cnt += 1

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

