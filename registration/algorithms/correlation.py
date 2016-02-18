""" Registration methods based on cross correlation """

from numpy import ndarray

from thunder.rdds.images import Images
from thunder.registration.registration import RegistrationMethod
from thunder.registration.methods.utils import computeDisplacement, computeReferenceMean, checkReference


class CrossCorr(RegistrationMethod):
    """
    Translation using cross correlation.
    """
    def __init__(self, *args, **kwargs):
        super(CrossCorr, self).__init__(*args, **kwargs)
        self.reference = None

    def prepare(self, images, startIdx=None, stopIdx=None, defaultNImages=20):
        """
        Prepare cross correlation by computing or specifying a reference image.

        `images` should be either a numpy ndarray or an Images object. If an ndarray
        is passed, it will be used as the reference; otherwise a reference image
        will be calculated as the mean of some subsection of the passed Images object.
        If startIdx or stopIdx are passed, then these will be used to determine
        the starting and ending points over which to calculate the reference mean.
        If neither startIdx nor stopIdx is passed, then the default behavior is to
        calculate a reference mean image over the center `defaultNImages` records
        of the Images object.

        Parameters
        ----------
        images : ndarray or Images object
            Images to compute reference from, or a single image to set as reference

        See computeReferenceMean.
        """
        if isinstance(images, Images):
            self.reference = computeReferenceMean(images, startIdx, stopIdx,
                                                  defaultNImages=defaultNImages)
        elif isinstance(images, ndarray):
            self.reference = images
        else:
            raise Exception('Must provide either an Images object or a reference')

        return self

    def isPrepared(self, images):
        """
        Check if cross correlation is prepared by checking the dimensions of the reference.

        See checkReference.
        """
        if self.reference is None:
            raise RuntimeError('Reference not defined; prepare() must be called before use.')
        checkReference(self.reference, images)

    def getTransform(self, im):
        """
        Compute displacement between an image or volume and reference.

        Displacements are computed using the dimensionality of the inputs,
        so will be 2D for images and 3D for volumes.

        Parameters
        ----------
        im : ndarray
            The image or volume
        """

        from thunder.registration.transformation import Displacement

        delta = computeDisplacement(im, self.reference)

        return Displacement(delta)