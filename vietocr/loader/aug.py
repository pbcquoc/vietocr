from PIL import Image
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia

class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    self.aug = iaa.Sequential(iaa.SomeOf((1, None), 
        [
        sometimes(iaa.Crop(percent=(0.01, 0.05), sample_independently=True)),
        sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
        sometimes(iaa.MotionBlur(k=3)),
        sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.01))),
        sometimes(iaa.KeepSizeByResize(iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1), fit_output=True))),
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.01))),
        sometimes(iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                            iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.25))])),

    ],
        random_order=True),
    random_order=True)
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    img = Image.fromarray(img)
    return img
