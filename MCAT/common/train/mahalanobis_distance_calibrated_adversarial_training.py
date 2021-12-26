import numpy
import torch
import common.torch
import common.summary
import common.numpy
import attacks
from .adversarial_training import *


class MahalanobisdistanceCalibratedAdversarialTraining(AdversarialTraining):


    def __init__(self, epochs, lambdaa, alpha, beta, wc, nogt, thres, model, trainset, testset, optimizer, scheduler, attack, objective, loss, fix, transition, fraction=0.5, augmentation=None, writer=common.summary.SummaryWriter(), cuda=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param attack: attack
        :type attack: attacks.Attack
        :param objective: objective
        :type objective: attacks.Objective
        :param loss: loss
        :type loss: callable
        :param loss: transition
        :type loss: callable
        :param fraction: fraction of adversarial examples per batch
        :type fraction: float
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert fraction < 1

        super(MahalanobisdistanceCalibratedAdversarialTraining, self).__init__(model, trainset, testset, optimizer, scheduler, attack, objective, fraction, augmentation, writer, cuda)

        self.loss = loss
        """ (callable) Loss. """

        self.total_step = epochs * len(self.trainset)
        self.lambdaa = lambdaa
        self.alpha = alpha
        self.beta = beta
        self.wc = wc
        self.fix = fix
        self.nogt = nogt
        self.thres = thres
        self.global_step = 0
        self.mdist = None

        if self.fix:
            self.optimizer1 = torch.optim.SGD(model.parameters(), lr=0.075, momentum=0.9)
            self.scheduler1 = common.train.get_exponential_scheduler(optimizer, batches_per_epoch=500, gamma=0.95)

        self.transition = transition
        """ (callable) Transition. """

        self.N_class = None
        """ (int) Number of classes. """

        if getattr(self.model, 'N_class', None) is not None:
            self.N_class = self.model.N_class

        if '__name__' in dir(self.loss):
            self.writer.add_text('config/loss', self.loss.__name__)
        else:
            self.writer.add_text('config/loss', self.loss.func.__name__)
        self.writer.add_text('config/transition', self.transition.__name__)

    def train(self, epoch0):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """
        if self.fix:
            epoch = epoch0 - 60
        else:
            epoch = epoch0
        if self.fix and epoch0 < 60:
            for b, (inputs, targets) in enumerate(self.trainset):
                if self.global_step < 0.1 * self.total_step:
                    weight = self.global_step * 10 / self.total_step
                    alpha = self.alpha * (0.01 * (1 - weight) + weight)
                    lambdaa = self.lambdaa * (0.1 * (1 - weight) + weight)
                    beta = 0
                    thres = self.thres * (0.1 * (1 - weight) + weight)
                else:
                    alpha = self.alpha
                    lambdaa = self.lambdaa
                    beta = 0
                    thres = self.thres

                if self.augmentation is not None:
                    inputs = self.augmentation.augment_images(inputs.numpy())

                inputs = common.torch.as_variable(inputs, self.cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, self.cuda)
                self.global_step += 1

                if self.N_class is None:
                    _ = self.model.forward(inputs)
                    self.N_class = _.size(1)

                # distributions = common.torch.one_hot(targets, self.N_class)

                split = int(self.fraction * inputs.size()[0])
                # update fraction for correct loss computation
                fraction = split / float(inputs.size(0))

                clean_inputs = inputs[:split]
                clean_targets = targets[:split]
                # clean_distributions = distributions[:split]
                # adversarial_distributions = distributions[split:]

                self.model.eval()

                inputs = clean_inputs

                self.model.train()
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                clean_logits = logits

                clean_loss = self.loss(clean_logits, 0, clean_targets, 0, alpha, lambdaa, beta, self.wc, self.nogt)
                clean_error = common.torch.classification_error(clean_logits, clean_targets)
                loss = clean_loss

                loss.backward()
                self.optimizer1.step()
                self.scheduler1.step()

                global_step = epoch * len(self.trainset) + b
                self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

                self.writer.add_scalar('train/loss', clean_loss.item(), global_step=global_step)
                self.writer.add_scalar('train/error', clean_error.item(), global_step=global_step)
                # self.writer.add_scalar('train/confidence', torch.mean(torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0]).item(), global_step=global_step)

                self.writer.add_histogram('train/logits', torch.max(clean_logits, dim=1)[0], global_step=global_step)
                # self.writer.add_histogram('train/confidences', torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0], global_step=global_step)

                if self.summary_gradients:
                    for name, parameter in self.model.named_parameters():
                        self.writer.add_histogram('train_weights/%s' % name, parameter.view(-1),
                                                  global_step=global_step)
                        self.writer.add_histogram('train_gradients/%s' % name, parameter.grad.view(-1),
                                                  global_step=global_step)

                self.writer.add_images('train/images', inputs[:min(16, split)], global_step=global_step)
                self.progress(epoch, b, len(self.trainset))
        else:
            if self.fix:
                logits = getattr(self.model, 'logits')
                logits.mu.requires_grad = False
            for b, (inputs, targets) in enumerate(self.trainset):
                if self.global_step < 0.1 * self.total_step:
                    weight = self.global_step * 10 / self.total_step
                    alpha = self.alpha * (0.01 * (1 - weight) + weight)
                    lambdaa = self.lambdaa * (0.1 * (1 - weight) + weight)
                    beta = self.beta * (0.01 * (1 - weight) + weight)
                    thres = self.thres * (0.1 * (1 - weight) + weight)
                else:
                    alpha = self.alpha
                    lambdaa = self.lambdaa
                    beta = self.beta
                    thres = self.thres

                if self.augmentation is not None:
                    inputs = self.augmentation.augment_images(inputs.numpy())

                inputs = common.torch.as_variable(inputs, self.cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, self.cuda)
                self.global_step += 1

                if self.N_class is None:
                    _ = self.model.forward(inputs)
                    self.N_class = _.size(1)

                #distributions = common.torch.one_hot(targets, self.N_class)

                split = int(self.fraction * inputs.size()[0])
                # update fraction for correct loss computation
                fraction = split / float(inputs.size(0))

                clean_inputs = inputs[:split]
                adversarial_inputs = inputs[split:]
                clean_targets = targets[:split]
                adversarial_targets = targets[split:]
                #clean_distributions = distributions[:split]
                #adversarial_distributions = distributions[split:]

                self.model.eval()
                self.objective.set(adversarial_targets)
                adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, adversarial_inputs, self.objective)
                adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
                adversarial_inputs = adversarial_inputs + adversarial_perturbations

                gamma, adversarial_norms = self.transition(adversarial_perturbations)
                #gamma = common.torch.expand_as(gamma, adversarial_distributions)

                #adversarial_distributions = adversarial_distributions*(1 - gamma)
                #adversarial_distributions += gamma*torch.ones_like(adversarial_distributions)/self.N_class

                inputs = torch.cat((clean_inputs, adversarial_inputs), dim=0)

                self.model.train()
                self.optimizer.zero_grad()
                logits = self.model(inputs)
                clean_logits = logits[:split]
                adversarial_logits = logits[split:]

                centers = getattr(self.model, 'logits').mu
                self.mdist = torch.sum(torch.sum(centers * centers, dim=1).unsqueeze(0) + torch.sum(centers * centers, dim=1).unsqueeze(1) - 2 * torch.matmul(centers, centers.permute(1,0))) / (self.N_class * (self.N_class - 1))
                dthres = thres * self.mdist
                if self.thres > 1.0:
                    dthres = dthres.detach()

                adversarial_loss = self.loss(adversarial_logits, gamma, adversarial_targets, dthres, alpha, lambdaa, beta, self.wc, self.nogt, clean_logits)
                adversarial_error = common.torch.classification_error(adversarial_logits, adversarial_targets)

                clean_loss = self.loss(clean_logits, 0, clean_targets, dthres, alpha, lambdaa, beta, self.wc, self.nogt)
                clean_error = common.torch.classification_error(clean_logits, clean_targets)
                loss = (1 - fraction) * clean_loss + fraction * adversarial_loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                global_step = epoch * len(self.trainset) + b
                self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

                self.writer.add_scalar('train/loss', clean_loss.item(), global_step=global_step)
                self.writer.add_scalar('train/error', clean_error.item(), global_step=global_step)
                #self.writer.add_scalar('train/confidence', torch.mean(torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0]).item(), global_step=global_step)

                self.writer.add_histogram('train/logits', torch.max(clean_logits, dim=1)[0], global_step=global_step)
                #self.writer.add_histogram('train/confidences', torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0], global_step=global_step)

                success = torch.clamp(torch.abs(adversarial_targets - torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[1]), max=1)
                self.writer.add_scalar('train/adversarial_loss', adversarial_loss.item(), global_step=global_step)
                self.writer.add_scalar('train/adversarial_error', adversarial_error.item(), global_step=global_step)
                #self.writer.add_scalar('train/adversarial_confidence', torch.mean(torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[0]).item(), global_step=global_step)
                self.writer.add_scalar('train/adversarial_success', torch.mean(success.float()).item(), global_step=global_step)

                self.writer.add_histogram('train/adversarial_logits', torch.max(adversarial_logits, dim=1)[0], global_step=global_step)
                #self.writer.add_histogram('train/adversarial_confidences', torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[0], global_step=global_step)

                self.writer.add_histogram('train/adversarial_objectives', adversarial_objectives, global_step=global_step)
                self.writer.add_histogram('train/adversarial_norms', adversarial_norms, global_step=global_step)

                if self.summary_gradients:
                    for name, parameter in self.model.named_parameters():
                        self.writer.add_histogram('train_weights/%s' % name, parameter.view(-1), global_step=global_step)
                        self.writer.add_histogram('train_gradients/%s' % name, parameter.grad.view(-1), global_step=global_step)

                self.writer.add_images('train/images', inputs[:min(16, split)], global_step=global_step)
                self.writer.add_images('train/adversarial_images', inputs[split:split + 16], global_step=global_step)
                self.progress(epoch, b, len(self.trainset))

    def test(self, epoch0):
        """
        Test on adversarial examples.

        :param epoch: epoch
        :type epoch: int
        """
        if self.fix:
            epoch = epoch0 - 60
        else:
            epoch = epoch0
        if self.fix and epoch0 < 60:
            self.model.eval()
            if self.mdist is None:
                centers = getattr(self.model, 'logits').mu
                self.mdist = torch.sum(
                    torch.sum(centers * centers, dim=1).unsqueeze(0) + torch.sum(centers * centers, dim=1).unsqueeze(
                        1) - 2 * torch.matmul(centers, centers.permute(1, 0))) / (self.N_class * (self.N_class - 1))
            # reason to repeat this here: use correct loss for statistics
            losses = None
            errors = None
            logits = None
            # confidences = None

            for b, (inputs, targets) in enumerate(self.testset):
                inputs = common.torch.as_variable(inputs, self.cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, self.cuda)

                if self.N_class is None:
                    _ = self.model.forward(inputs)
                    self.N_class = _.size(1)

                # distributions = common.torch.as_variable(common.torch.one_hot(targets, self.N_class))

                outputs = self.model(inputs)
                losses = common.numpy.concatenate(losses, self.loss(outputs, 0, targets, self.beta * self.mdist, 0,
                                                                    self.lambdaa, self.beta, self.wc, self.nogt,
                                                                    reduction='none').detach().cpu().numpy())
                errors = common.numpy.concatenate(errors, common.torch.classification_error(outputs, targets,
                                                                                            reduction='none').detach().cpu().numpy())
                logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                # confidences = common.numpy.concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
                self.progress(epoch, b, len(self.testset))

            global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1
            self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
            self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
            self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
            # self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)

            self.writer.add_histogram('test/losses', losses, global_step=global_step)
            self.writer.add_histogram('test/errors', errors, global_step=global_step)
            self.writer.add_histogram('test/logits', logits, global_step=global_step)
            # self.writer.add_histogram('test/confidences', confidences, global_step=global_step)
        else:
            self.model.eval()
            if self.mdist is None:
                centers = getattr(self.model, 'logits').mu
                self.mdist = torch.sum(
                    torch.sum(centers * centers, dim=1).unsqueeze(0) + torch.sum(centers * centers, dim=1).unsqueeze(
                        1) - 2 * torch.matmul(centers, centers.permute(1, 0))) / (self.N_class * (self.N_class - 1))

            # reason to repeat this here: use correct loss for statistics
            losses = None
            errors = None
            logits = None
            #confidences = None

            for b, (inputs, targets) in enumerate(self.testset):
                inputs = common.torch.as_variable(inputs, self.cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, self.cuda)

                if self.N_class is None:
                    _ = self.model.forward(inputs)
                    self.N_class = _.size(1)

                #distributions = common.torch.as_variable(common.torch.one_hot(targets, self.N_class))

                outputs = self.model(inputs)
                losses = common.numpy.concatenate(losses, self.loss(outputs, 0, targets, self.beta * self.mdist, 0, self.lambdaa, self.beta, self.wc, self.nogt, reduction='none').detach().cpu().numpy())
                errors = common.numpy.concatenate(errors, common.torch.classification_error(outputs, targets, reduction='none').detach().cpu().numpy())
                logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                #confidences = common.numpy.concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
                self.progress(epoch, b, len(self.testset))

            global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1
            self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
            self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
            self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
            #self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)

            self.writer.add_histogram('test/losses', losses, global_step=global_step)
            self.writer.add_histogram('test/errors', errors, global_step=global_step)
            self.writer.add_histogram('test/logits', logits, global_step=global_step)
            #self.writer.add_histogram('test/confidences', confidences, global_step=global_step)

            self.model.eval()

            losses = None
            errors = None
            logits = None
            #confidences = None
            successes = None
            norms = None
            objectives = None

            for b, (inputs, targets) in enumerate(self.testset):
                if b >= self.max_batches:
                    break

                inputs = common.torch.as_variable(inputs, self.cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                outputs_clean = self.model(inputs)
                targets = common.torch.as_variable(targets, self.cuda)
                #distributions = common.torch.as_variable(common.torch.one_hot(targets, self.N_class))

                self.objective.set(targets)
                adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, inputs, self.objective)
                objectives = common.numpy.concatenate(objectives, adversarial_objectives)

                adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
                inputs = inputs + adversarial_perturbations

                gamma, adversarial_norms = self.transition(adversarial_perturbations)
                #gamma = common.torch.expand_as(gamma, distributions)
                #distributions = distributions * (1 - gamma) + gamma * torch.ones_like(distributions) / self.N_class

                outputs = self.model(inputs)
                losses = common.numpy.concatenate(losses, self.loss(outputs, gamma, targets, self.beta * self.mdist, 0, self.lambdaa, self.beta, self.wc, self.nogt, outputs_clean, reduction='none').detach().cpu().numpy())
                errors = common.numpy.concatenate(errors, common.torch.classification_error(outputs, targets, reduction='none').detach().cpu().numpy())
                logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                #confidences = common.numpy.concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
                successes = common.numpy.concatenate(successes, torch.clamp(torch.abs(targets - torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[1]), max=1).detach().cpu().numpy())
                norms = common.numpy.concatenate(norms, adversarial_norms.detach().cpu().numpy())
                self.progress(epoch, b, self.max_batches)

            global_step = epoch + 1# * len(self.trainset) + len(self.trainset) - 1
            self.writer.add_scalar('test/adversarial_loss', numpy.mean(losses), global_step=global_step)
            self.writer.add_scalar('test/adversarial_error', numpy.mean(errors), global_step=global_step)
            self.writer.add_scalar('test/adversarial_logit', numpy.mean(logits), global_step=global_step)
            #self.writer.add_scalar('test/adversarial_confidence', numpy.mean(confidences), global_step=global_step)
            self.writer.add_scalar('test/adversarial_norm', numpy.mean(norms), global_step=global_step)
            self.writer.add_scalar('test/adversarial_objective', numpy.mean(objectives), global_step=global_step)
            self.writer.add_scalar('test/adversarial_success', numpy.mean(successes), global_step=global_step)

            self.writer.add_histogram('test/adversarial_losses', losses, global_step=global_step)
            self.writer.add_histogram('test/adversarial_errors', errors, global_step=global_step)
            self.writer.add_histogram('test/adversarial_logits', logits, global_step=global_step)
            #self.writer.add_histogram('test/adversarial_confidences', confidences, global_step=global_step)
            self.writer.add_histogram('test/adversarial_norms', norms, global_step=global_step)
            self.writer.add_histogram('test/adversarial_objectives', objectives, global_step=global_step)