from PIL import Image
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia

class ImgAugTransform:
  def __init__(self):
    self.aug = iaa.Sequential([
        iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        iaa.Sometimes(0.25, iaa.MotionBlur(k=5)),
        iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
#        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1)),
#        iaa.PiecewiseAffine(scale=(0.01, 0.01)),
        iaa.Sometimes(0.25,
                      iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                 iaa.CoarseDropout(0.1, size_percent=0.5)])),

    ])
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    img = Image.fromarray(img)
    return img
