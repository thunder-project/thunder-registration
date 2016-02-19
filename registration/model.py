from numpy import asarray
from .utils import check_images

class RegistrationModel(object):
    """
    A registration model, defined as a dictionary of transformations, one per image
    """
    def __init__(self, transformations, algorithm=None):
        self.transformations = transformations
        self.algorithm = algorithm

    def __getitem__(self, entry):
        return self.transformations[entry]

    def toarray(self):
        """
        Return transformations as an array with shape (n,x1,x2,...)
        where n is the number of images, and remaining dimensions depend
        on the particular transformations
        """
        return asarray([x.toarray() for x in self.transformations.values()])

    def transform(self, images):
        """
        Apply the transformation to an Images object.

        Will apply the underlying dictionary of transformations to
        the images or volumes of the Images object. The dictionary acts as a lookup
        table specifying which transformation should be applied to which record of the
        Images object based on the key. Because transformations are small,
        we broadcast the transformations rather than using a join.
        """
        images = check_images(images)

        def apply(item):
            (k, v) = item
            return self.transformations[k].apply(v)

        return images.map(apply, with_keys=True)

    def __repr__(self):
        s = self.__class__.__name__
        s += '\nlength: %g' % len(self.transformations)
        s += '\nalgorithm: ' + self.algorithm
        return s


class Transformation(object):
    """ 
    Base class for transformations 
    """
    def apply(self, im):
        raise NotImplementedError

    def toarray(self):
        raise NotImplementedError        


class Displacement(Transformation):
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

    def toarray(self):
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

    @staticmethod
    def compute(a, b):
        """
        Compute an optimal displacement between two ndarrays.

        Finds the displacement between two ndimensional arrays. Arrays must be
        of the same size. Algorithm uses a cross correlation, computed efficiently
        through an n-dimensional fft.

        Parameters
        ----------
        a : ndarray
            The first array

        b : ndarray
            The second array
        """
        from numpy.fft import rfftn, irfftn
        from numpy import unravel_index, argmax

        # compute real-valued cross-correlation in fourier domain
        s = a.shape
        f = rfftn(a)
        f *= rfftn(b).conjugate()
        c = abs(irfftn(f, s))

        # find location of maximum
        inds = unravel_index(argmax(c), s)

        # fix displacements that are greater than half the total size
        pairs = zip(inds, a.shape)
        # cast to basic python int for serialization
        adjusted = [int(d - n) if d > n // 2 else int(d) for (d, n) in pairs]

        return Displacement(adjusted)

    def __repr__(self):
        return "Displacement(delta=%s)" % repr(self.delta)
