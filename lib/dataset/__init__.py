from chainer import datasets
from . import binary_tree
from . import wordnet
from . import breakout
from lib.dataset import mnist_activity
import math

def get_data_shape(dataset_type, **kwargs):
    if dataset_type == 'synthetic':
        return math.ceil(2 ** kwargs['depthReal'] - 1)
    elif dataset_type == 'mnist':
        return kwargs['mnistShape']
    elif dataset_type == 'breakout':
        return (1, 80, 80)
    elif dataset_type == 'wordnet':
        return 82115
    elif dataset_type == 'mammal':
        return 1181
    else:
        raise NotImplementedError


def get_dataset(dataset_type, matrixForData, **kwargs):

    if dataset_type == 'synthetic':
        train = binary_tree.get_data(matrixForData)
        valid = train.copy()
        test_data = train.copy()
        if kwargs['dataset_randomness'] != -1:
            train = binary_tree.ProbabilisticBinaryTreeDataset(
                train, eps=kwargs['dataset_randomness'])
            valid = binary_tree.ProbabilisticBinaryTreeDataset(
                valid, eps=kwargs['dataset_randomness'])
            test = binary_tree.ProbabilisticBinaryTreeDataset(
                test_data, eps=kwargs['dataset_randomness'])

    elif dataset_type == 'mnist':
        # Load the MNIST dataset
        ndim = kwargs.get('ndim') if 'ndim' in kwargs else 1
        train, test = mnist_activity.get_mnist(withlabel=False, ndim=ndim, data=matrixForData, dtype=matrixForData.dtype)

#        train, test = datasets.get_mnist(withlabel=False, ndim=ndim)

        # Binarize dataset
        #train[train >= 0.5] = 1.0
        #train[train < 0.5] = 0.0
        #test[test >= 0.5] = 1.0
        #test[test < 0.5] = 0.0

        size_data = len(train[:, 1])
        upper_part = math.floor(0.8 * size_data)

        train, valid = datasets.split_dataset(train, upper_part)

    elif dataset_type == 'cifar100':
        # Load the Cifar-100 dataset
        train, test = datasets.get_cifar100(withlabel=False)
        train = 2 * (train - 0.5)
        test = 2 * (test - 0.5)

        train, valid = datasets.split_dataset(train, 49000)

    elif dataset_type == 'breakout':
        train, test = breakout.load_dataset(withlabel=False)
        # scaling data from [0, 1] to [-1, 1]
        train = 2 * (train - 0.5)
        test = 2 * (test - 0.5)
        train, valid = datasets.split_dataset(train, 80000)

    elif dataset_type == 'wordnet':
        num_negatives = kwargs['num_negatives']
        symmetrize = kwargs['symmetrize']
        assert num_negatives == 1
        train = wordnet.load_dataset(num_negatives, symmetrize)
        valid = None
        test = None

    elif dataset_type == 'mammal':
        num_negatives = kwargs['num_negatives']
        symmetrize = kwargs['symmetrize']
        assert num_negatives == 1
        train = wordnet.load_dataset(num_negatives, symmetrize, mammal=True)
        valid = None
        test = None

    else:
        raise ValueError

    return train, valid, test
