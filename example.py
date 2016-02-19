from numpy import arange
from scipy.ndimage.interpolation import shift

from registration import CrossCorr

reference = arange(9).reshape(3, 3)
deltas = [[1, 0], [0, 1]]
shifted = [shift(reference, delta, mode='wrap', order=0) for delta in deltas]

register = CrossCorr()
model = register.fit(shifted, reference=reference)

print model.transformations