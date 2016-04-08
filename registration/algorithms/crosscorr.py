from numpy import ndarray, asarray

from ..utils import check_images, check_reference
from ..model import RegistrationModel
from ..transforms import Displacement, LocalDisplacement


class CrossCorr(object):
    """
    Registration using cross correlation.

    Attributes
    ----------
    axis : integer
        An axis to restrict displacements to, e.g. use axis=2 to
        restrict displacements to axes (0, 1)
    """
    def __init__(self, axis=None):
        self.axis = axis

    def _get(self, image, reference):
        if self.axis is not None:
            return LocalDisplacement.compute(image, reference, self.axis)
        else:
            return Displacement.compute(image, reference)

    def fit(self, images, reference=None):
        """
        Estimate registration model using cross-correlation.

        Use cross correlation to compute displacements between 
        images or volumes and reference. Displacements will be 
        2D for images and 3D for volumes.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.

        reference : array-like
            A reference image to align to.
        """
        images = check_images(images)
        reference = check_reference(images, reference)

        def func(item):
            key, image = item
            return asarray([key, self._get(image, reference)])

        transformations = images.map(func, with_keys=True).toarray()
        if images.shape[0] == 1: 
            transformations = [transformations]

        algorithm = self.__class__.__name__
        return RegistrationModel(dict(transformations), algorithm=algorithm)

    def fit_and_transform(self, images, reference=None):
        """
        Estimate and apply registration model using cross-correlation.

        Use cross correlation to compute displacements between 
        images or volumes and reference, and apply the
        estimated model to the data. Displacements will be 
        2D for images and 3D for volumes.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.

        reference : array-like
            A reference image to align to.
        """
        images = check_images(images)
        check_reference(images, reference)

        def func(image):
            t = self._get(image, reference)
            return t.apply(image)

        return images.map(func)