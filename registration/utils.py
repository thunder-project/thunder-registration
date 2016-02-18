""" Transformations produced by registration methods """
from numpy import asarray
from thunder.utils.serializable import Serializable


class Transformation(object):
    """ Base class for transformations """

    def apply(self, im):
        raise NotImplementedError


class Displacement(Transformation, Serializable):
    """
    Class for transformations based on spatial displacements.

    Can be applied to either images or volumes.

    Parameters
    ----------
    delta : list
        A list of spatial displacements for each dimensino,
        e.g. [10,5,2] for a displacement of 10 in x, 5 in y, 2 in z
    """

    def __init__(self, delta=None):
        self.delta = delta

    def toArray(self):
        """
        Return transformation as an array
        """
        return asarray(self.delta)

    def apply(self, im):
        """
        Apply an n-dimensional displacement by shifting an image or volume.

        Parameters
        ----------
        im : ndarray
            The image or volume to shift
        """
        from scipy.ndimage.interpolation import shift

        return shift(im, map(lambda x: -x, self.delta), mode='nearest')

    def __repr__(self):
        return "Displacement(delta=%s)" % repr(self.delta)
