# thunder-registration

Algorithms for registering sequences of images. Includes a collection of `algorithms` that can be `fit` to data, all of which return a `model` that can be used to `transform` new data, in the `scikit-learn` style. Built on `numpy` and `scipy`. Works well alongside `thunder` and supprts parallelization, but can be used as a standalone module on local arrays.

## installation

```bash
pip install thunder-registration
```

## example

Create shifted copies of a reference image

```python
from numpy import arange
from scipy.ndimage.interpolation import shift

reference = arange(9).reshape(3, 3)
deltas = [[1, 0], [0, 1]]
shifted = [shift(reference, delta, mode='wrap', order=0) for delta in deltas]
```

Then perform registration using cross correlation

```python
from registration import CrossCorr

register = CrossCorr()
model = register.fit(shifted, reference=reference)
```

The estimated transformations should match the `deltas` we used

```python
print(model.transformations)
>> {(0,): Displacement(delta=[1, 0]), (1,): Displacement(delta=[0, 1])}
```

## usage

First pick an algorithm. Some algorithms take parameters, some don't.

```python
from registration import CrossCorr
algorithm = CrossCorr()
```

Fit the algorithm to compute registration parameters and return a model

```python
from registration import CrossCorr
model = algorithm.fit(data)
```

The attribute `model.transformations` is a dictionary mapping image index to whatever transformation type was returned by the fitting. You can use the model to apply the estimated registration to the same or different data.

```python
registered = model.transform(data)
```

Or do both at once

```python
registered = algorithm.fit_and_transform(data)
```

You can also save and load models (TODO)

```python
model.save('model.json')
```

```python
from registration import load
model = load('model.json')
```

## algorithms

##### `CrossCorr(axis=None).fit(images, reference)`

Uses cross-correlation to estimate an integer n-dimensional displacement between all images and a reference.

- `axis` specify an axis to restrict estimates to e.g. `axis=2` to only estimate displacements in (0,1)
- `images` an ndarray or thunder images object with the images to align
- `reference` an ndarray reference image

## tests

Run tests with 

```
py.test
```

Tests run locally by default, but the exact same tests can be run against Spark locally using

```
py.test --engine=spark
```