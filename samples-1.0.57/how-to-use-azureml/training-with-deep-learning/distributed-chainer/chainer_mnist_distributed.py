
import argparse

import chainer
import chainer.cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import chainermn
import chainermn.datasets
import chainermn.functions


chainer.disable_experimental_feature_warning = True


class MLP0SubA(chainer.Chain):
    def __init__(self, comm, n_out):
        super(MLP0SubA, self).__init__(
            l1=L.Linear(784, n_out))

    def __call__(self, x):
        return F.relu(self.l1(x))


class MLP0SubB(chainer.Chain):
    def __init__(self, comm):
        super(MLP0SubB, self).__init__()

    def __call__(self, y):
        return y


class MLP0(chainermn.MultiNodeChainList):
    # Model on worker 0.
    def __init__(self, comm, n_out):
        super(MLP0, self).__init__(comm=comm)
        self.add_link(MLP0SubA(comm, n_out), rank_in=None, rank_out=1)
        self.add_link(MLP0SubB(comm), rank_in=1, rank_out=None)


class MLP1Sub(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP1Sub, self).__init__(
            l2=L.Linear(None, n_units),
            l3=L.Linear(None, n_out))

    def __call__(self, h0):
        h1 = F.relu(self.l2(h0))
        return self.l3(h1)


class MLP1(chainermn.MultiNodeChainList):
    # Model on worker 1.
    def __init__(self, comm, n_units, n_out):
        super(MLP1, self).__init__(comm=comm)
        self.add_link(MLP1Sub(n_units, n_out), rank_in=0, rank_out=0)


def main():
    parser = argparse.ArgumentParser(
        description='ChainerMN example: pipelined neural network')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    if args.gpu:
        comm = chainermn.create_communicator('hierarchical')
        data_axis, model_axis = comm.rank % 2, comm.rank // 2
        data_comm = comm.split(data_axis, comm.rank)
        model_comm = comm.split(model_axis, comm.rank)
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator('naive')
        data_axis, model_axis = comm.rank % 2, comm.rank // 2
        data_comm = comm.split(data_axis, comm.rank)
        model_comm = comm.split(model_axis, comm.rank)
        device = -1

    if model_comm.size != 2:
        raise ValueError(
            'This example can only be executed on the even number'
            'of processes.')

    if comm.rank == 0:
        print('==========================================')
        if args.gpu:
            print('Using GPUs')
        print('Num unit: {}'.format(args.unit))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    if data_axis == 0:
        model = L.Classifier(MLP0(model_comm, args.unit))
    elif data_axis == 1:
        model = MLP1(model_comm, args.unit, 10)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()
        model.to_gpu()

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), data_comm)
    optimizer.setup(model)

    # Original dataset on worker 0 and 1.
    # Datasets of worker 0 and 1 are split and distributed to all workers.
    if model_axis == 0:
        train, test = chainer.datasets.get_mnist()
        if data_axis == 1:
            train = chainermn.datasets.create_empty_dataset(train)
            test = chainermn.datasets.create_empty_dataset(test)
    else:
        train, test = None, None
    train = chainermn.scatter_dataset(train, data_comm, shuffle=True)
    test = chainermn.scatter_dataset(test, data_comm, shuffle=True)

    train_iter = chainer.iterators.SerialIterator(
        train, args.batchsize, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, data_comm)
    trainer.extend(evaluator)

    # Some display and output extentions are necessary only for worker 0.
    if comm.rank == 0:
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
