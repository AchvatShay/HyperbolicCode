
import chainer

import numpy
import math

def get_mnist(withlabel=True, ndim=1, scale=1., dtype=None,
              label_dtype=numpy.int32, rgb_format=False, data=None):
    """Gets the MNIST dataset.

    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ is a set of hand-written
    digits represented by grey-scale 28x28 images. In the original images, each
    pixel is represented by one-byte unsigned integer. This function
    scales the pixels to floating point values in the interval ``[0, scale]``.

    This function returns the training set and the test set of the official
    MNIST dataset. If ``withlabel`` is ``True``, each dataset consists of
    tuples of images and labels, otherwise it only consists of images.

    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        ndim (int): Number of dimensions of each image. The shape of each image
            is determined depending on ``ndim`` as follows:

            - ``ndim == 1``: the shape is ``(784,)``
            - ``ndim == 2``: the shape is ``(28, 28)``
            - ``ndim == 3``: the shape is ``(1, 28, 28)``

        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).
        label_dtype: Data type of the labels.
        rgb_format (bool): if ``ndim == 3`` and ``rgb_format`` is ``True``, the
            image will be converted to rgb format by duplicating the channels
            so the image shape is (3, 28, 28). Default is ``False``.

    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    dtype = chainer.get_dtype(dtype)
    train_raw = data.astype(numpy.float64)

    mean_value = numpy.mean(train_raw, axis=1)
    std_value = numpy.std(train_raw, axis=1)

    train_raw = numpy.divide(train_raw.T - mean_value,  std_value)

    train_raw = train_raw.T
    #train_raw *= scale / max_value
    #train_raw[train_raw < 0] = 0.0

    size_data = len(train_raw[:, 1])
    i = range(math.ceil(size_data / 4))

    test_raw = train_raw[i, :]

    return train_raw, test_raw