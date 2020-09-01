#!/usr/bin/env python

import pathlib

from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from lib import models, dataset

from lib.miscs.hyperparams import Hyperparams
from lib.miscs import pathorganizer
from lib.miscs.utils import assetsdir, datadir

from lib.miscs.logutils import logger_manager
logger = logger_manager.register(__name__)

TEST_BATCH_SIZE = 50

def get_model(hpt):
    data_shape = dataset.get_data_shape(hpt.dataset.type, **hpt.dataset)
    encoder = models.make_encoder(
        hpt.model.type, hpt.model.p_z, data_shape=data_shape, **hpt.model)
    decoder = models.make_decoder(
        hpt.model.type, hpt.model.p_x, data_shape=data_shape, **hpt.model)
    prior = models.make_prior(hpt.model.type, hpt.model.p_z, **hpt.model)

    avg_elbo_loss = models.AvgELBOLoss(
        encoder, decoder, prior, beta=hpt.loss.beta, k=hpt.loss.k)
    return avg_elbo_loss


def evaluate(hpt, train, test, avg_elbo_loss):
    if hpt.dataset.type in ('mnist', 'breakout'):
        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):

            test_iter = chainer.iterators.SerialIterator(
                test, TEST_BATCH_SIZE, repeat=False, shuffle=False)

            logger.info('calculate test ELBO/LL')
            test_k = 500
            test_elbo = 0
            test_ll = 0
            for batch in tqdm(test_iter,
                              total=int(np.ceil(len(test) / TEST_BATCH_SIZE))):
                test_elbo_, test_ll_ = avg_elbo_loss.get_elbo(
                    avg_elbo_loss.xp.asarray(batch), k=test_k, with_ll=True)
                test_elbo += test_elbo_.array
                test_ll += test_ll_.array
            test_elbo /= np.ceil(len(test) / TEST_BATCH_SIZE)
            test_ll /= np.ceil(len(test) / TEST_BATCH_SIZE)
        return {
            'Test AvgELBO': float(test_elbo),
            'Test LL': float(test_ll),
        }

    elif hpt.dataset.type == 'synthetic':
        from lib.dataset import binary_tree
        from lib import functions

        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            z = avg_elbo_loss.encoder(train.data).mean
            if hpt.dataset.dataset_randomness != -1:
                z_prob = avg_elbo_loss.encoder(np.array(train[:])).mean

        N = len(train.data)
        hamming_dists = np.zeros((N, N))
        euclid_dists = np.zeros((N, N))
        z_dists = np.zeros((N, N))
        z_dists_prob = np.zeros((N, N))
        if hpt.model.p_z == 'nagano':
            comparator = lambda x, y: functions.lorentz_distance(
                x[None, :], y[None, :]).array[0]
        else:
            comparator = lambda x, y: np.sqrt(((x.array - y.array) ** 2).sum())

        for i in range(len(train)):
            for j in range(i):
                hamming_dists[i, j] = binary_tree.hamming_distance(
                    train.data[i], train.data[j])
                euclid_dists[i, j] = binary_tree.euclid_distance(
                    train.data[i], train.data[j])
                z_dists[i, j] = comparator(z[i], z[j])
                z_dists_prob[i, j] = comparator(z_prob[i], z_prob[j])

        treeShortesFile = loadmat(hpt.dataset['matrixFile_struct'])
        testSP = treeShortesFile.get('matrixForSP')

        filt_ = np.fromfunction(lambda i, j: i > j, shape=hamming_dists.shape)
        return {
            'corr-with-SP': np.corrcoef(
                testSP[filt_], z_dists[filt_])[0, 1],
            'corr-with-hamming': np.corrcoef(
                hamming_dists[filt_], z_dists[filt_])[0, 1],
            'corr-with-euclid': np.corrcoef(
                euclid_dists[filt_], z_dists[filt_])[0, 1],
            'corr-with-hamming-noise': np.corrcoef(
                hamming_dists[filt_], z_dists_prob[filt_])[0, 1],
            'corr-with-euclid-noise': np.corrcoef(
                euclid_dists[filt_], z_dists_prob[filt_])[0, 1]
        }

    else:
        raise AttributeError


def visualize(hpt, train, test, avg_elbo_loss, treeFile):
    import matplotlib.pyplot as plt
    from lib.miscs import plot_utils

    if hpt.dataset.type in ('mnist', 'breakout'):
        def save_images(x, filename):
            fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
            for ai, xi in zip(ax.flatten(), x):
                if hpt.dataset.type == 'mnist':
                    ai.imshow(xi.reshape(28, 28))
                elif hpt.dataset.type == 'breakout':
                    ai.imshow(xi.reshape(80, 80) / 2 + 0.5)
            fig.savefig(filename)

        avg_elbo_loss.to_cpu()
        train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
        x = chainer.Variable(np.asarray(train[train_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = avg_elbo_loss.decoder(
                avg_elbo_loss.encoder(x).mean, n_batch_axes=1).mean
        save_images(x.array, (po.imagesdir() / 'train').as_posix())
        save_images(
            x1.array, (po.imagesdir() / 'train_reconstructed').as_posix())

        test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
        x = chainer.Variable(np.asarray(test[test_ind]))
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            x1 = avg_elbo_loss.decoder(
                avg_elbo_loss.encoder(x).mean, n_batch_axes=1).mean
        save_images(x.array, (po.imagesdir() / 'test').as_posix())
        save_images(
            x1.array, (po.imagesdir() / 'test_reconstructed').as_posix())

        # draw images from randomly sampled z
        z = avg_elbo_loss.prior().sample(9)
        x = avg_elbo_loss.decoder(z, n_batch_axes=1).mean
        save_images(x.array, (po.imagesdir() / 'sampled').as_posix())

    elif hpt.dataset.type == 'synthetic':
        from lib.dataset import binary_tree
        from lib import functions
        from scipy.cluster import hierarchy

        with chainer.using_config('train', False), \
                chainer.using_config('enable_backprop', False):
            z = avg_elbo_loss.encoder(train.data).mean
            if hpt.dataset.dataset_randomness != -1:
                z_prob = avg_elbo_loss.encoder(np.array(train[:])).mean

        if hpt.model.p_z == 'euclid':
            z_vis = z.array
            if hpt.dataset.dataset_randomness != -1:
                z_prob_vis = z_prob.array
        elif hpt.model.p_z == 'nagano':
            z_vis = functions.h2p(z).array
            if hpt.dataset.dataset_randomness != -1:
                z_prob_vis = functions.h2p(z_prob).array
        else:
            raise NotImplementedError

        z_distance_l = np.zeros((len(z.array), len(z.array)))
        for i in range(len(z.array)):
            for j in range(len(z.array)):
                if i == j :
                    z_distance_l[i, j] = 0
                else :
                    z_distance_l[i, j] = functions.lorentz_distance(z.array[i, :], z.array[j, :]).item()

        z_distance_p = np.zeros((len(z_vis), len(z_vis)))
        for i in range(len(z_vis)):
            for j in range(len(z_vis)):
                if i == j:
                    z_distance_p[i, j] = 0
                else:
                    z_distance_p[i, j] = functions.poincare_distance(z_vis[i, :], z_vis[j, :], -1).item()

        mdic = {"loranzDistMat": z_distance_l, "poincareDistMat": z_distance_p}
        savemat((pathlib.WindowsPath(hpt.general.outputpath) / f'matlab_matrix{hpt.model.p_x}_{hpt.model.n_hidden}_{hpt.training.iteration}.mat').as_posix(), mdic)
        savemat((po.namedir() / f'matlab_matrix{hpt.model.p_x}_{hpt.model.n_hidden}_{hpt.training.iteration}.mat').as_posix(), mdic)

        plot_utils.getfig((4, 4))

        train_tmp = train.data
        train_tmp[train_tmp != 0] = 1

        ax = plt.axes(projection='3d')

        for i in range(len(train.data)):
            for j in range(i):
                if binary_tree.hamming_distance(
                        train.data[i], train.data[j]) == 1:
                    ax.plot(
                        [z_vis[i, 0], z_vis[j, 0]], [z_vis[i, 1], z_vis[j, 1]], [z_vis[i, 2], z_vis[j, 2]],
                        lw=1, color='gray', zorder=1)

        for i in range(len(z_vis)):
            ax.scatter(
                z_vis[i, 0], z_vis[i, 1], z_vis[i, 2], s=(300 / train_tmp[i, :].sum(axis=-1) ** 2),
                c='#F15A29', zorder=10)
            ax.text(z_vis[i, 0], z_vis[i, 1], z_vis[i, 2], treeFile.get('points_name')[i], size=5, zorder=10,
                    color='k')

        ax.scatter(0, 0, 0, s=100, c='magenta', zorder=20, marker='x')
        plt.pause(1e-13)
        plt.savefig(
            (po.imagesdir() / 'embedding.pdf').as_posix(),
            bbox_inches='tight')
        plt.savefig(
            (po.imagesdir() / 'embedding.png').as_posix(),
            bbox_inches='tight', dpi=400)

        plot_utils.getfig((4, 4))

        for i in range(len(train.data)):
            for j in range(i):
                if binary_tree.hamming_distance(
                        train.data[i], train.data[j]) == 1:
                    plt.plot(
                        [z_vis[i, 0], z_vis[j, 0]], [z_vis[i, 1], z_vis[j, 1]],
                        lw=1, color='gray', zorder=1)

        for i in range(len(z_vis)):
            plt.scatter(
                z_vis[i, 0], z_vis[i, 1], s=(300 / train_tmp[i, :].sum(axis=-1) ** 2),
                c='#F15A29', zorder=10)
            plt.text(z_vis[i, 0], z_vis[i, 1], treeFile.get('points_name')[i], size=8, zorder=10,
                    color='k')

        plt.scatter(0, 0, s=100, c='magenta', zorder=20, marker='x')
        plt.pause(1e-13)
        plt.savefig(
            (po.imagesdir() / 'embedding_2d.pdf').as_posix(),
            bbox_inches='tight')
        plt.savefig(
            (po.imagesdir() / 'embedding_2d.png').as_posix(),
            bbox_inches='tight', dpi=400)
    else:
        raise NotImplementedError


def main(hpt):
    logger.info('load New Data From Matlab')
    if hpt.dataset.type == 'synthetic':
        treeFile = loadmat(hpt.dataset['matrixFile_struct'])
        matrixForData = treeFile.get('matrixForHS')
        hpt.dataset.__setitem__('depth', treeFile.get('depthToSave')[0, 0])
        hpt.dataset.__setitem__('depthReal', treeFile.get('depthToSaveReal')[0, 0])
        hpt.training.__setitem__('batch_size', treeFile.get('baches')[0, 0])
        print(treeFile.get('matrixForHS'))
        print(hpt.dataset.depth)
        print(hpt.training.batch_size)
    elif hpt.dataset.type == 'mnist':
        activityFile = loadmat(hpt.dataset['matrixFile_activity'])
        matrixForData = np.transpose(activityFile.get('roiActivity'))
        hpt.dataset.__setitem__('mnistShape', len(matrixForData[1, :]))
        hpt.training.__setitem__('batch_size', len(matrixForData[:, 1]))

    logger.info('build model')
    avg_elbo_loss = get_model(hpt)
    if hpt.general.gpu >= 0:
        avg_elbo_loss.to_gpu(hpt.general.gpu)

    logger.info('setup optimizer')
    if hpt.optimizer.type == 'adam':
        optimizer = chainer.optimizers.Adam(alpha=hpt.optimizer.lr)
    optimizer.setup(avg_elbo_loss)

    logger.info('load dataset')
    train, valid, test = dataset.get_dataset(hpt.dataset.type, matrixForData, **hpt.dataset)

    if hpt.general.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        valid, _ = chainer.datasets.split_dataset(valid, 100)
        test, _ = chainer.datasets.split_dataset(test, 100)

    train_iter = chainer.iterators.SerialIterator(
        train, hpt.training.batch_size)
    valid_iter = chainer.iterators.SerialIterator(
        valid, hpt.training.batch_size, repeat=False, shuffle=False)

    logger.info('setup updater/trainer')
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        device=hpt.general.gpu, loss_func=avg_elbo_loss)

    if not hpt.training.early_stopping:
        trainer = training.Trainer(
            updater, (hpt.training.iteration, 'iteration'),
            out=po.namedir(output='str'))
    else:
        trainer = training.Trainer(
            updater, triggers.EarlyStoppingTrigger(
                monitor='validation/main/loss',
                patients=5,
                max_trigger=(hpt.training.iteration, 'iteration')
            ), out=po.namedir(output='str'))

    if hpt.training.warm_up != -1:
        time_range = (0, hpt.training.warm_up)
        trainer.extend(
            extensions.LinearShift('beta', value_range=(0.1, hpt.loss.beta),
                                   time_range=time_range,
                                   optimizer=avg_elbo_loss))

    trainer.extend(extensions.Evaluator(
        valid_iter, avg_elbo_loss, device=hpt.general.gpu))
    # trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(
        extensions.snapshot_object(
            avg_elbo_loss, 'avg_elbo_loss_snapshot_iter_{.updater.iteration}'),
        trigger=(int(hpt.training.iteration / 5), 'iteration'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/reconstr', 'main/kl_penalty', 'main/beta', 'lr', 'elapsed_time'
    ]))
    trainer.extend(extensions.ProgressBar())

    logger.info('run training')
    trainer.run()

    logger.info('save last model')
    extensions.snapshot_object(
        avg_elbo_loss, 'avg_elbo_loss_snapshot_iter_{.updater.iteration}'
    )(trainer)

    logger.info('evaluate')
    metrics = evaluate(hpt, train, test, avg_elbo_loss)
    for metric_name, metric in metrics.items():
        logger.info('{}: {:.4f}'.format(metric_name, metric))

    if hpt.general.noplot:
        return metrics

    logger.info('visualize images')
    visualize(hpt, train, test, avg_elbo_loss, treeFile)

    return metrics


if __name__ == '__main__':
    chainer.global_config.dtype= np.float64

    current_file = pathlib.Path(__file__)
    hpt = Hyperparams(current_file.parents[1] / 'recipes/mlp_default.yml')
    hpt = hpt.parse_args(program_name=current_file.name)

    logger.info(hpt.summary())

    po = pathorganizer.PathOrganizer(
        root=pathlib.WindowsPath(hpt.general.outputpath), name=hpt.general.name, datadir=datadir)

    with (po.namedir() / 'option.yml').open('w') as f:
        hpt.dump(f)

    if hpt.general.num_experiments == 1:
        main(hpt)
    else:
        metrics = []
        for experiment_idx in range(hpt.general.num_experiments):
            logger.info('Trial [{}/{}] start'.format(
                experiment_idx + 1, hpt.general.num_experiments))
            if experiment_idx > 0:
                hpt['general']['noplot'] = True
            metrics.append(main(hpt))
        metrics = pd.DataFrame(metrics)
        logger.info('Total result:')
        for name in metrics.columns:
            logger.info('{}: {:.4f} \\pm {:.4f}'.format(
                name, metrics[name].mean(), metrics[name].std()))
        metrics.to_csv((po.logsdir() / 'metrics.tsv').as_posix(), sep='\t')
