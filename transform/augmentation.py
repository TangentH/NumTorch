import numpy as np
from skimage import transform, util
import random
from common.tensor import tensor

def _rotate(image, angle):
    return transform.rotate(image, angle)

def _scale(image, scale_factor):
    return transform.rescale(image, scale_factor, anti_aliasing=True)

def _translate(image, translation):
    return transform.warp(image, transform.AffineTransform(translation=translation))

def _shear(image, shear_factor):
    return transform.warp(image, transform.AffineTransform(shear=shear_factor))

def _add_noise(image, mode='gaussian'):
    return util.random_noise(image, mode=mode)

def augment_image(images):
    augmented_images = []
    if images.ndim == 1:
        images = [images]
    for image in images:
        image_raw = image.copy()
        if image.ndim == 1:
            shape = np.sqrt(image.size).astype(int)
            image = image.reshape(shape, shape)
        
        image = image / 255.0
        
        angle = random.uniform(-30, 30)
        image = _rotate(image, angle)
        
        # scale_factor = random.uniform(0.8, 1.0)
        # image = _scale(image, scale_factor)
        
        translation = (random.uniform(-2, 2), random.uniform(-2, 2))
        image = _translate(image, translation)
        
        shear_factor = random.uniform(-0.1, 0.1)
        image = _shear(image, shear_factor)
        
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        image = transform.resize(image, (shape, shape), anti_aliasing=True)
        
        if image_raw.ndim == 1:
            image = image.flatten()
        
        augmented_images.append(image)
    
    return tensor(np.array(augmented_images))
