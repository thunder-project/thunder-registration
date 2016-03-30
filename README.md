# thunder-registration

> algorithms for registering sequences of images

This package Includes a collection of algorithms for image registration. It is well-suited to registering movies obtained in the medical or neuroscience imaging domains, but can be applied to any image sequences requiring alignment.

The API is designed around `algorithms` that can be `fit` to data, all of which return a `model` that can be used to `transform` new data, in the style of [`scikit-learn`](http://scikit-learn.org/stable/). Built on [`numpy`](https://github.com/numpy/numpy) and [`scipy`](https://github.com/scipy/scipy). Compatible with Python 2.7+ and 3.4+. Works well alongside [`thunder`](https://github.com/thunder-project/thunder) and supprts parallelization via [`spark`](https://github.com/apache/spark), but can be used as a standalone package on local [`numpy`](https://github.com/numpy/numpy) arrays.

## installation

```bash
pip install thunder-registration
```

## example

In this example we create shifted copies of a reference image and then align them

```python
# create shifted copies

from numpy import arange
from scipy.ndimage.interpolation import shift
reference = arange(9).reshape(3, 3)
deltas = [[1, 0], [0, 1]]
shifted = [shift(reference, delta, mode='wrap', order=0) for delta in deltas]

# perform registration

from registration import CrossCorr
register = CrossCorr()
model = register.fit(shifted, reference=reference)

# the estimated transformations should match the deltas we used

print(model.transformations)
>> {(0,): Displacement(delta=[1, 0]), (1,): Displacement(delta=[0, 1])}
```

## usage

Import and construct an algorithm.

```python
from registration import CrossCorr
algorithm = CrossCorr()
```

Fit the algorithm to `data` to compute registration parameters and return a model

```python
model = algorithm.fit(data)
```

The attribute `model.transformations` is a dictionary mapping image index to whatever transformation type was returned by the fitting. You can use the model to apply the estimated registration to the same or different data.

```python
registered = model.transform(data)
```

## api

### algorithm

All algorithms have the following methods:

#### `algorithm.fit(data, opts)`

Fits the algorthm to the `data`, with optional arguments depending on the algorithm. The `data` can be a [`numpy`](https://github.com/numpy/numpy) `ndarray` or a [`thunder`](https://github.com/thunder-project/thunder) `Images` object.

### model

The result of fitting an `algorithm` to data is a `model`.

A `model` has the following properties and methods:

#### `model.transformations`

A dictionary mapping image index to the transformation returned by fitting.

#### `model.transform(data)`

Applies the estimated transformations to new data. As with fitting, `data` can be a [`numpy`](https://github.com/numpy/numpy) `ndarray` or a [`thunder`](https://github.com/thunder-project/thunder) `Images` object.

## list of algorithms

The following algorithms are available:

#### `CrossCorr(axis=None).fit(images, reference)`

Uses cross-correlation to estimate an integer n-dimensional displacement between all images and a reference.

- `axis` specify an axis to restrict estimates to e.g. `axis=2` to only estimate displacements in (0,1)
- `images` an ndarray or thunder images object with the images to align
- `reference` an ndarray reference image

## tests

Run tests with 

```
py.test
```

Tests run locally by default, but the same tests can be run against Spark locally using

```
py.test --engine=spark
```
