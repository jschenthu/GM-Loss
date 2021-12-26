import torch
import numpy
import common.torch
import common.numpy
import common.summary
from common.log import log


def progress(batch, batches, epoch=None):
    """
    Report progress.

    :param epoch: epoch
    :type epoch: int
    :param batch: batch
    :type batch: int
    :param batches: batches
    :type batches: int
    """

    if batch == 0:
        if epoch is not None:
            log(' %d .' % epoch, end='')
        else:
            log(' .', end='')
    else:
        log('.', end='', context=False)

    if batch == batches - 1:
        log(' done', end="\n", context=False)


def test(model, testset, perturbations=None, cuda=False, return_labels=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    :param return_labels: whether to additionally return labels
    :type return_labels: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    probabilities = None
    labels = None
    likelihoods = None
    errors = None

    # should work with and without labels
    start = 0
    for b, data in enumerate(testset):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0].cuda()
            labels_ = data[1].cuda()
        else:
            inputs = data

        if return_labels is True:
            assert len(data) >= 2
            labels = common.numpy.concatenate(labels, data[1].numpy())

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        inputs = inputs.permute(0, 3, 1, 2)
        if perturbations is not None:
            #print(perturbations.shape, inputs.shape)
            inputs += perturbations[start: start + inputs.shape[0]]
            start += inputs.shape[0]

        logits = model(inputs)
        preds_ = torch.max(logits, dim=1)[1]
        errors_ = (preds_ != labels_).long().cpu().numpy()
        probabilities_ = torch.nn.functional.softmax(logits, dim=1)
        probabilities_ = torch.max(probabilities_, dim=1)[0].detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)
        dists_b = (logits * (-2))
        #dists_b = dists_b.scatter(1, labels_.unsqueeze(1), torch.max(dists_b, dim=1, keepdim=True)[0])
        likelihoods_ = torch.min(dists_b, dim=1)[0].detach().cpu().numpy()
        likelihoods = common.numpy.concatenate(likelihoods, likelihoods_)
        errors = common.numpy.concatenate(errors, errors_)

        progress(b, len(testset))

    assert probabilities.shape[0] == len(testset.dataset)

    if return_labels:
        return probabilities, likelihoods, errors
    else:
        return probabilities, likelihoods, errors


def attack(model, testset, attack, objective, attempts=1, writer=common.summary.SummaryWriter(), cuda=False):
    """
    Attack model.
    
    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: attack objective
    :type objective: attacks.Objective
    :param attempts: number of attempts
    :type attempts: int
    :param writer: summary writer or utility function to get writer
    :type writer: torch.utils.tensorboard.SummaryWriter or callable
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert attempts >= 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    perturbations = []
    probabilities = []
    likelihoods = []
    errors = []
    #print('Start attacking')
    #print(len(testset))

    # should work via subsets of datasets
    for a in range(attempts):
        perturbations_a = None
        probabilities_a = None
        likelihoods_a = None
        errors_a = None

        for b, data in enumerate(testset):
            assert isinstance(data, tuple) or isinstance(data, list)

            inputs = common.torch.as_variable(data[0], cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            labels = common.torch.as_variable(data[1], cuda)

            # attack target labels
            targets = None
            if len(list(data)) > 2:
                targets = common.torch.as_variable(data[2], cuda)

            objective.set(labels, targets)
            perturbations_b, errors_b = attack.run(model, inputs, objective,
                                                   writer=writer if not callable(writer) else writer('%d-%d' % (a, b)),
                                                   prefix='%d/%d/' % (a, b) if not callable(writer) else '')

            inputs = inputs + common.torch.as_variable(perturbations_b, cuda)
            logits = model(inputs)
            probabilities_b = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
            dists_b = (logits * (-2))
            dists_b = dists_b.scatter(1, labels.unsqueeze(1), torch.max(dists_b, dim=1, keepdim=True)[0])
            likelihoods_b = torch.min(dists_b, dim=1)[0].detach().cpu().numpy()

            perturbations_a = common.numpy.concatenate(perturbations_a, perturbations_b)
            probabilities_a = common.numpy.concatenate(probabilities_a, probabilities_b)
            likelihoods_a = common.numpy.concatenate(likelihoods_a, likelihoods_b)
            errors_a = common.numpy.concatenate(errors_a, errors_b)

            progress(b, len(testset), epoch=a)

        perturbations.append(perturbations_a)
        probabilities.append(probabilities_a)
        likelihoods.append(likelihoods_a)
        errors.append(errors_a)

    perturbations = numpy.array(perturbations)
    probabilities = numpy.array(probabilities)
    likelihoods = numpy.array(likelihoods)
    errors = numpy.array(errors)

    assert perturbations.shape[1] == len(testset.dataset)
    assert probabilities.shape[1] == len(testset.dataset)
    assert likelihoods.shape[1] == len(testset.dataset)
    assert errors.shape[1] == len(testset.dataset)

    return perturbations, probabilities, likelihoods, errors
    # attempts x N x C x H x W, attempts x N x K, attempts x N
