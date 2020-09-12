from PIL import Image
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia

class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    self.aug = iaa.Sequential([
#        sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
        sometimes(iaa.MotionBlur(k=3)),
        sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
#        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        sometimes(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1))),
#        iaa.PiecewiseAffine(scale=(0.01, 0.01)),
        sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])),

    ])
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    img = Image.fromarray(img)
    return img
