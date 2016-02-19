# thunder-registration

Algorithms for registering sequences of images. Includes a collection of `algorithms` that can be `fit` to data, all of which return a `model` that can be used to `transform` new data, in the `scikit-learn` style. Built on `numpy` and `scipy`. Works well alongside `thunder` and supprts parallelization, but can be used as a standalone module on local arrays.

# installation

```bash
pip install thunder-registration
```

# example

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

### usage

Run an algorithm to compute registration parameters

```python
from registration import CrossCorr
model = CrossCorr().fit(data)
```

Use a model to apply the estimated registration to the same or different data

```python
registered = model.transform(data)
```

Or do both at once

```python
registered = model.fit_and_transform(data)
```

Save and load models

```python
model.save('model.json')
```

```python
from registration import load
model = load('model.json')
```

### algorithms

##### `CrossCorr().fit(images, reference)`

Uses cross-correlation to estimate an integer n-dimensional displacement between all images and a reference.

- `images` an ndaraay or thunder images object with the images to align
- `reference` an ndarray reference image
