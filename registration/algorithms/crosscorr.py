from numpy import ndarray

from ..utils import check_images, check_reference
from ..model import RegistrationModel, Displacement


class CrossCorr(object):
    """
    Registration using cross correlation.
    """
    def __init__(self):
        pass

    def fit(self, images, reference=None):
        """
        Perform registration by computing integer
        displacements between an image or volume and reference
        using cross correlation.

        Displacements will be 2D for images and 3D for volumes.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.

        reference : array-like
            A reference image to align to.
        """
        images = check_images(images)
        check_reference(images, reference)

        func = lambda image: Displacement.compute(image, reference)
        transformations = images.map_generic(func, return_dict=True)

        algorithm = self.__class__.__name__
        return RegistrationModel(transformations, algorithm=algorithm)

    def fit_and_transform(self, images, reference=None):
        images = check_images(images)
        check_reference(images, reference)

        def func(image):
            t = Displacement.compute(image, reference)
            return t.apply(image)

        return images.map(func)

