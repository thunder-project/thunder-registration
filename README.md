# thunder-registration

Algorithms for registering sequences of images. Includes a collection of `algorithms` that can be `fit` to data, all of which return a `model` that can be used to `transform` new data, in the `scikit-learn` style. Built on `numpy`, `scipy`, `sklearn`, and `skimage`. Works well alongside `thunder`, but can be used as a standalone module on local arrays.

# installation

```bash
pip install thunder-registration
```

# examples

### algorithms

run an algorithm to compute registration parameters

```python
from registration import CrossCorr
model = CrossCorr(params).fit(data)
```

then align an image sequence using those parameters

```python
registered = model.transform(data)
```

or do both at once

```python
registered = model.run(data)
```

### models

loading

```python
from register import load
model = load('model.json')
result = model.transform(data)
```

saving

```python
model.save('model.json')
```
