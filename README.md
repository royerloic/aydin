# aydin

Denoising, but chill...

## Status
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![codecov](https://codecov.io/gl/aydinorg/aydin/branch/master/graph/badge.svg?token=gV3UqFAg5U)](https://codecov.io/gl/aydinorg/aydin)

## Installation

You can get the suitable aydin executable from [here]() or you 
can do `python setup.py develop`, for mac first need to do `brew install libomp`.

## How to run?

For now CLI can be only run as a module:

To run a Noise2Self:
```bash
aydin noise2self path/to/noisyimage
```

To run a Noise2Truth:
```bash
aydin noise2truth path/to/noisyimage path/to/truthimage path/to/testimage
```

## Recommended Environment

#### Linux

- Ubuntu 18.04+

#### Windows

- Windows 10

#### OSX

- Mojave 10.14.6+
