import torch
from torch import optim, autograd

import darts.functional as F


class Hyperparameters:
    alpha_lr = 3e-4
    alpha_wd = 1e-3


class Architecture:

    def __init__(self, model, args, hyperparams=Hyperparameters(), device='cpu'):
        self.momentum = args.momentum  # momentum for optimizer of theta
        self.wd = args.weight_decay  # weight decay for optimizer of model's theta
        self.model = model  # main model with respect to theta and alpha
        self.device = device

        # this is the optimizer to optimize alpha parameter
        self.optimizer = optim.Adam(
            self.model.arch_parameters(),
            lr=hyperparams.alpha_lr,
            betas=(0.5, 0.999),
            weight_decay=hyperparams.alpha_wd
        )

    def comp_unrolled_model(self, data, target, eta, optimizer):
        """ Loss on train set and then update w_pi, not-in-place

        Parameters
        ----------
        data : torch.tensor

        target : torch.tensor
        eta : float
        optimizer : torch.optim.optimizer
             optimizer of theta, not optimizer of alpha

        Returns
        -------
        model_unrolled
        """
        # forward to get loss
        loss = self.model.loss(data, target)
        # flatten current weights
        theta = F.flatten(self.model.parameters()).detach()
        try:
            # fetch momentum data from theta optimizer
            moment = F.flatten(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)
        except Exception:
            moment = torch.zeros_like(theta)

        # flatten all gradients
        dtheta = F.flatten(autograd.grad(loss, self.model.parameters())).data
        # indeed, here we implement a simple SGD with momentum and weight decay
        # theta = theta - eta * (moment + weight decay + dtheta)
        theta = theta.sub(eta, moment + dtheta + self.wd * theta)
        # construct a new model
        unrolled_model = self.construct_model_from_theta(theta)

        return unrolled_model.to(self.device)

    def step(self, x_train, target_train, x_valid, target_valid, eta, optimizer, unrolled):
        """
        update alpha parameter by manually computing the gradients
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta:
        :param optimizer: theta optimizer
        :param unrolled:
        :return:
        """
        # alpha optimizer
        self.optimizer.zero_grad()

        # compute the gradient and write it into tensor.grad
        # instead of generated by loss.backward()
        if unrolled:
            self.backward_step_unrolled(x_train, target_train, x_valid, target_valid, eta, optimizer)
        else:
            # directly optimize alpha on w, instead of w_pi
            self.backward_step(x_valid, target_valid)

        self.optimizer.step()

    def backward_step(self, x_valid, target_valid):
        """
        simply train on validate set and backward
        :param x_valid:
        :param target_valid:
        :return:
        """
        _, loss = self.model.loss(x_valid, target_valid, reduce='mean')
        # both alpha and theta require grad but only alpha optimizer will
        # step in current phase.
        loss.backward()

    def backward_step_unrolled(self, x_train, target_train, x_valid, target_valid, eta, optimizer):
        """
        train on validate set based on update w_pi
        :param x_train:
        :param target_train:
        :param x_valid:
        :param target_valid:
        :param eta: 0.01, according to author's comments
        :param optimizer: theta optimizer
        :return:
        """
        # theta_pi = theta - lr * grad
        unrolled_model = self.comp_unrolled_model(x_train, target_train, eta, optimizer)
        # calculate loss on theta_pi
        unrolled_loss = unrolled_model.loss(x_valid, target_valid)

        # this will update theta_pi model, but NOT theta model
        unrolled_loss.backward()
        # grad(L(w', a), a), part of Eq. 6
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self.hessian_vector_product(vector, x_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            # g = g - eta * ig, from Eq. 6
            g.data.sub_(eta, ig.data)

        # write updated alpha into original model
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)

    def construct_model_from_theta(self, theta):
        """
        construct a new model with initialized weight from theta
        it use .state_dict() and load_state_dict() instead of
        .parameters() + fill_()
        :param theta: flatten weights, need to reshape to original shape
        :return:
        """
        model = self.model.new()
        state_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = v.numel()
            # restore theta[] value to original shape
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        state_dict.update(params)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def hessian_vector_product(self, vector, data, target, r=1e-2):
        """
        slightly touch vector value to estimate the gradient with respect to alpha
        refer to Eq. 7 for more details.
        :param vector: gradient.data of parameters theta
        :param x:
        :param target:
        :param r:
        :return:
        """
        R = r / F.flatten(vector).norm()

        for p, v in zip(self.model.parameters(), vector):
            # w+ = w + R * v
            p.data.add_(R, v)

        loss = self.model.loss(data, target)
        # gradient with respect to alpha
        grads_p = autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # w- = (w+R*v) - 2R*v
            p.data.sub_(2 * R, v)

        loss = self.model.loss(data, target)
        grads_n = autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            # w = (w+R*v) - 2R*v + R*v
            p.data.add_(R, v)

        h = [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
        # h len: 2 h0 torch.Size([14, 8])
        # print('h len:', len(h), 'h0', h[0].shape)
        return h