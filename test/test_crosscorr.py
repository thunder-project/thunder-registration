import pytest
from numpy import arange, allclose
from scipy.ndimage.interpolation import shift

from registration import CrossCorr

pytestmark = pytest.mark.usefixtures("eng")

def test_fit(eng):
	reference = arange(25).reshape(5, 5)
	algorithm = CrossCorr()
	deltas = [[1, 2], [-2, 1]]
	shifted = [shift(reference, delta, mode='wrap', order=0) for delta in deltas]
	model = algorithm.fit(shifted, reference=reference)
	assert allclose(model.toarray(), deltas)


def test_fit_3d(eng):
	reference = arange(125).reshape(5, 5, 5)
	algorithm = CrossCorr()
	deltas = [[1, 0, 2], [0, 1, 2]]
	shifted = [shift(reference, delta, mode='wrap', order=0) for delta in deltas]
	model = algorithm.fit(shifted, reference=reference)
	assert allclose(model.toarray(), deltas)