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
        return asarray([value.toarray() for (key, value) in sorted(self.transformations.items())])

    def transform(self, images):
        """
        Apply the transformation to an Images object.

        Will apply the underlying dictionary of transformations to
        the images or volumes of the Images object. The dictionary acts as a lookup
        table specifying which transformation should be applied to which record of the
        Images object based on the key. Because transformations are small,
        we broadcast the transformations rather than using a join.

        Parameters
        ----------
        images : array-like or thunder images
            The sequence of images / volumes to register.
        """
        images = check_images(images)

        def apply(item):
            (k, v) = item
            return self.transformations[k].apply(v)

        return images.map(apply, value_shape=images.value_shape, dtype=images.dtype, with_keys=True)

    def __repr__(self):
        s = self.__class__.__name__
        s += '\nlength: %g' % len(self.transformations)
        s += '\nalgorithm: ' + self.algorithm
        return s
