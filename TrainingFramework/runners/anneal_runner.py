import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import models.small as small
import models.vnet as vnet
import models.deepvnet as deepvnet
from PIL import Image
import runners.pytorch_helpers as phelp

__all__ = ['AnnealRunner']

def binarize(structs, means=None):
    """
    This method discretizes a microstructure (maps it to
    0 and 1) while enforcing the mean.

    This is done following a congruent method to what is 
    used in the post-processing from the previous paper. 

    ############################################
    STRUCTURE MUST BE THE FORM (#xSPATIAL DIMENSIONS)
        - can be 2D or 3D
    ############################################
    """
    structs = structs.detach().numpy()

    if means is None:
        means = np.mean(structs, axis=tuple(range(1, len(structs.shape))))
    elif type(means) is float:
        means = np.ones((len(structs),)) * means

    N = np.product(structs.shape[1:])

    new_structures = np.zeros_like(structs)

    for n, (struct, mean) in enumerate(zip(structs, means)):
        index = np.unravel_index(np.argpartition(struct, -int(N * mean),
                             axis=None)[-int(N * mean):], struct.shape)

        new_structures[n][index] = 1.0

        #print(f"SBG Mean: {struct.mean()}. Desired Mean: {mean}.")
        #f, ax = plt.subplots(1, 3, figsize=[12, 5])
        #ax[0].imshow(struct, cmap='gray')
        #ax[1].imshow(new_structures[n], cmap='gray')
        #ax[2].imshow(np.abs(struct - new_structures[n]), cmap='gray')
        #plt.show()

    return torch.from_numpy(new_structures).float()

def train_test_split(data, fraction=0.8):
    from random import choice
    order = torch.randperm(len(data))
    cutoff = int(len(data) * fraction)
    train = data[order[:cutoff], ...]
    test = data[order[cutoff:], ...]
    return train, test

class constantLR(object):
    def step(self):
        pass

class exponential_lambda(object):
    """
    This class implements an exponential decay cycling function. 

    it is meant to be used in conjunction with a cyclic learning rate
    under a cycle setting with a step_size_up of 1. 
    """
    def __init__(self, decay_length):
        """
        :decay_length: The number of steps in each decay cycle. 
        """
        self.cycle=0
        self.factor = 0.05 ** (2 / decay_length) # most of the decay should take half the length
        self.multiplier = 1.0

    def __call__(self, x):
        if x != self.cycle:
            self.multiplier = 1.0
            self.cycle = x

        temp = self.multiplier
        self.multiplier *= self.factor
        return temp

class AnnealRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_model(self):
        # THE FOLLOWING MODELS WERE CONSIDERED DURING DEVELOPMENT
        # BUT WERE NOT USED IN THE FINAL PROPOSED FRAMEWORK. WE INCLUDE
        # THEM HERE SIMPLY FOR PEOPLE'S INTEREST. 
        
        # THE RECOMMENDED MODELS ARE MARKED BELOW.
        if self.config.model.model.lower() == 'conlain':
            return vnet.Conlain
        elif self.config.model.model.lower() == 'inception':
            return small.Inception
        elif self.config.model.model.lower() == 'attnincept':
            return small.AttentionInception
        elif self.config.model.model.lower() == 'vnetci':
            return small.VNetC_Inception
        elif self.config.model.model.lower() == 'rvnetci':
            return small.Residual_VNetC_Inception
        elif self.config.model.model.lower() == 'vnet1':
            return vnet.VNet1
        elif self.config.model.model.lower() == 'vnet2':
            return vnet.VNet2
        elif self.config.model.model.lower() == 'vnet3':
            return vnet.VNet3
        elif self.config.model.model.lower() == 'vnetcia':
            return small.VNetC_Inception_InitialAttn
        elif self.config.model.model.lower() == 'vnetciia':
            return small.VNetC_Inception_Attn
        elif self.config.model.model.lower() == 'rvnetcia':
            return small.Residual_VNetC_Inception_InitialAttn
        elif self.config.model.model.lower() == 'rvnetciia':
            return small.Residual_VNetC_Inception_Attn
        elif self.config.model.model.lower() == 'vnetc':
            return small.VNetC
        elif self.config.model.model.lower() == 'vnetca':
            return small.VNetC_Attn

        # THE FOLLOWING ARE THE RECOMMENDED MODELS.
        # vnetd2 was used in Case Study 1.
        # vnetd4 was used in Case Study 2.
        elif self.config.model.model.lower() == 'vnetd2':
            return deepvnet.VNetD2
        elif self.config.model.model.lower() == 'vnetd3':
            return deepvnet.VNetD3
        elif self.config.model.model.lower() == 'vnetd4':
            return deepvnet.VNetD4
        else:
            raise NotImplementedError(f"{self.config.model.model} is not supported.")

    def get_optimizer(self, parameters):
        
        if self.config.optim.optimizer.lower() == 'adam':
            return optim.Adam(params=parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer.lower() == 'adamw':
            return optim.AdamW(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer.lower() == 'rmsprop':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer.lower() == 'sgd':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=self.config.optim.beta1)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def get_swa_model(self, model):
        """
        Returns the SWA (Stochastic Weighted Averaging)
        averaged model. 
        """
        if self.config.optim.swa.lower() == 'ema':
            # Exponential Moving Average (recommended by Song)
            def EMA(averaged_model_parameter, model_parameter, num_averaged):
                if num_averaged == 1:
                    # the very first update. Just take the model.
                    return model_parameter
                else:
                    return self.config.optim.ema_param * averaged_model_parameter + \
                            (1 - self.config.optim.ema_param) * model_parameter

            averaged_model = optim.swa_utils.AveragedModel(model, \
                    avg_fn = EMA
                    )
            return averaged_model

        elif self.config.optim.swa.lower() == 'swa':
            # standard SWA using Equal Running Average
            return optim.swa_utils.AveragedModel(model)
        else:
            raise NotImplementedError(f"{self.config.optim.swa} is not supported.")

    def get_swalr_scheduler(self, optimizer):
        """
        Returns the learning rate scheduler associated with the
        Stochastic Weighted Averaging / EMA section of the training
        run. 

        Moves to the stationary learning rate (swalr parameter) in 
        1 / 3 the total number of remaining steps. 
        """
        assert self.config.optim.swa.lower() != 'none'
        num_steps = int(self.config.training.n_iters * \
                (1-self.config.optim.start_percentage) * 0.33)

        return optim.swa_utils.SWALR(optimizer,
                                     anneal_strategy = 'cos',
                                     anneal_epochs = num_steps,
                                     swa_lr = self.config.optim.swalr,
                                     )

    def get_lr_scheduler(self, optimizer):
        """ returns the learning rate scheduler. """
        if self.config.optim.scheduler.lower() == 'constant':
            return constantLR()
        elif self.config.optim.scheduler.lower() == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(optimizer, \
                    max_lr = self.config.optim.lr, \
                    total_steps = self.config.training.n_iters+1, \
                    pct_start = self.config.optim.pct_start, \
                    cycle_momentum = False, #self.config.optim.cycle_momentum, \
                    #base_momentum = 0.85, \
                    #max_momentum = self.config.optim.beta1, \
                    div_factor = self.config.optim.initial_div, \
                    final_div_factor = self.config.optim.final_div, \
                    )
        elif self.config.optim.scheduler.lower() == 'lr_test':
            return optim.lr_scheduler.LinearLR(optimizer, \
                    start_factor=1e-6, \
                    end_factor=1.0, \
                    total_iters=self.config.optim.growthsteps, \
                    )

        elif self.config.optim.scheduler.lower() == 'expcycling':
            # cycling with exponential decay (decay occurs in about
            # in about half the cyclic width)
            return optim.lr_scheduler.CyclicLR(optimizer, 
                                base_lr = self.config.optim.lr,
                                max_lr = self.config.optim.max_lr,
                                step_size_up = 1.0,
                                step_size_down = self.config.optim.cycle,
                                scale_fn = exponential_lambda(
                                            self.config.optim.cycle
                                                             ),
                                scale_mode = 'cycle',
                                cycle_momentum = False,
                                )
        elif self.config.optim.scheduler.lower() == 'cosine':
            # implements cosine annealing. But there is no decay going on here.
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                T_max = self.config.optim.cycle,
                                eta_min = self.config.optim.max_lr,
                                )
        else:
            raise NotImplementedError(f"LR Scheduler {self.config.optim.scheduler} not understood")



    def logit_transform(self, image, lam=1e-6):
        """
        I've just hijacked their code to go from -1 to 1
        """
        # their original code below:
        #image = lam + (1 - 2 * lam) * image
        #return torch.log(image) - torch.log1p(-image)
        return image * 2 - 1

    def train(self):
        # transformations. 
        if self.config.data.random_flip is False and self.config.data.dataset == 'NBSA':
            # I don't want it to change the data at all. I have curated it
            # how it should be. 
            tran_transform = test_transform = lambda x: x
        else:
            raise NotImplementedError(f"Unsupported dataset: {self.config.data.dataset}, usng 'NBSA'.")
        
        # load the dataset
        if self.config.data.dataset == 'NBSA':
            # checking if its a torch tensor or an h5py file:
            if self.config.data.data_location.__contains__('.h5'):
                # its an h5 file:
                dataset, test_dataset = phelp.train_test_split_h5py(\
                        self.config.data.data_location, 'micros', \
                        fraction=self.config.data.test_train_fraction)

                y = torch.rand(len(dataset), 1)
                y_test = torch.rand(len(test_dataset), 1)
                dataset.add(y)
                test_dataset.add(y_test)

            elif self.config.data.data_location.__contains__('.pth'):
                data = torch.load(self.config.data.data_location).float()
                if len(data.shape) != 4:
                    data = data.unsqueeze(1)
                data, test = train_test_split(data, fraction=self.config.data.test_train_fraction) # I changed this

                y = torch.rand(data.shape[0], 1)
                y_test = torch.rand(test.shape[0], 1)

                dataset = TensorDataset(data, y)
                test_dataset = TensorDataset(test, y_test)
            else:
                raise NotImplementedError("This file type is not supported.")
        else:
            raise NotImplementedError(f"Unsupported dataset: {self.config.data.dataset}, usng 'NBSA'.")


        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 drop_last=True)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # initialization the model
        deep_model = self.get_model()
        score = deep_model(self.config).to(self.config.device)

        # initialization:
        if self.config.model.initialize.lower() == 'default':
            pass
        elif self.config.model.initialize.lower() == 'kaiming_fan_in':
            for layer in score.modules():
                if type(layer) is nn.Conv2d:
                    init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
        elif self.config.model.initialize.lower() == 'xavier':
            for layer in score.modules():
                if type(layer) is nn.Conv2d:
                    init.xavier_normal_(layer.weight, gain=2**(1/2))
        else:
            raise AttributeError("{self.config.model.initialize} is not an implemented initialize parameter.")
        # end initialization

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        # SWA: Check if we want any type of Stochastic Weighted Averaging
        swa_flag = True if self.config.optim.swa.lower() != 'none' else False
        if swa_flag:
            swa_model = self.get_swa_model(score)
            swalr_scheduler = self.get_swalr_scheduler(optimizer)
            assert not self.args.resume_training, "Please don't restart SWA training."

        # resuming training. 
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

            # check about freezing weights... this only really applies
            # if training is resumed (e.g.,we are transfering)
            if self.config.training.n_frozen_layers > 0:
                # we are freezing layers. 
                # (starts from deepest to shallowest)
                for i in range(1, min(self.config.training.n_frozen_layers + 1, 5)):
                    # ResNet
                    getattr(score.module, f"res{int(5-i)}").requires_grad_(False)
                    # RefineNet Block
                    getattr(score.module, f"refine{i}").requires_grad_(False)


        # learning rate scheduling:
        lr_scheduler = self.get_lr_scheduler(optimizer)
        if self.args.resume_training:
            assert self.config.optim.scheduler.lower() == 'constant', \
                    "Only constant scheduling can be restarted."

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)


        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                #print(type(X))
                #print(X.dtype)
                #raise AttributeError
                step += 1
                score.train()
                X = X.to(self.config.device)
                #X = X / 256. * 255. + torch.rand_like(X) / 256.
                # centered uniform dequantization (i really doubt this matters)
                X = X + (2 * torch.rand_like(X) - 1) * self.config.model.sigma_end / 2
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)
                else:
                    raise NotImplementedError(f"{self.config.training.algo} is not supported.")

                optimizer.zero_grad()
                loss.backward()

                # -------------------------------------------------------
                # gradient clipping
                if self.config.optim.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(score.parameters(), \
                            self.config.optim.grad_clip)
                # -------------------------------------------------------

                optimizer.step()

                # scheduling
                if swa_flag:
                    if step > (self.config.optim.start_percentage * \
                            self.config.training.n_iters):
                        # update the SWA model
                        swa_model.update_parameters(score)

                        # update the learning rate
                        swalr_scheduler.step()
                    else:
                        lr_scheduler.step()
                else:
                    lr_scheduler.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    #test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    test_X = test_X + (2 * torch.rand_like(test_X) - 1) * self.config.model.sigma_end / 2
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    if True: #self.config.training.algo == 'dsm':
                        with torch.no_grad():
                            test_loss = anneal_dsm_score_estimation(score, test_X, test_labels, \
                                    sigmas, self.config.training.anneal_power)
                            test_lab = 'test_dsm_loss'
                        tb_logger.add_scalar(test_lab, test_loss, global_step=step)
                        logging.info("TEST: step: {}, loss: {}".format(step, test_loss.item()))

                    elif self.config.training.algo == 'ssm':
                        " Its not working and I don't want to deal with it."
                        test_loss = anneal_sliced_score_estimation_vr(score, test_X, test_labels, \
                                sigmas, \
                                n_particles=self.config.training.n_particles)
                        test_lab = 'test_ssm_loss'


                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

                    if swa_flag:
                        torch.save(swa_model.state_dict(), os.path.join(self.args.log, 'swa_checkpoint_{}.pth'.format(step)))
                        torch.save(swa_model.state_dict(), os.path.join(self.args.log, 'swa_checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def anneal_Langevin_dynamics_NOCLAMP(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        """ The same as the annealed langevin dynamics code, except that no clamping is done. """

        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    if self.config.data.logit_transform:
                        # we need to transform the data
                        images.append(0.5 * (x_mod.to('cpu') + 1))
                    else:
                        images.append(x_mod.to('cpu'))

                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)

                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    if self.config.data.logit_transform:
                        # we need to transform the data
                        images.append(torch.clamp(0.5 * (x_mod + 1), 0.0, 1.0).to('cpu'))
                    else:
                        images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))

                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def anneal_Langevin_dynamics_conditional_small(self, x_mod, scorenet, \
            sigmas, n_steps_each=100, step_lr=0.00002, starting_index=0):
        images = []
        images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))

        sigmas = sigmas[starting_index:]
        # perterb the input based on the noise level it should be at. 
        x_mod = x_mod + torch.randn_like(x_mod) * sigmas[0]

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                c = c + starting_index
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise

                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))

            return images

    def anneal_Langevin_dynamics_conditional_whole(self, x_mod, scorenet, sigmas, \
            n_steps_each=100, step_lr=0.00002, starting_index=2):
        """
        This method is a annealed langevin dynamics solver where
        the starting point is not random noise, but an image. 
        The solver is not run from the max to min noise level,
        but rather from a starting_index to the minimum noise
        level. 

        Additionally, the whole microstructure is run through the model
        """
        model_size = self.config.data.image_size
        sigmas = sigmas[starting_index:]

        print(f'n_steps_each: {n_steps_each}.')

        # perterb the input based on the noise level it should be at. 
        x_mod = x_mod + torch.randn_like(x_mod) * sigmas[0]

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), \
                    desc='annealed Langevin dynamics sampling'):
                # adjusting c:
                c = c + starting_index

                # creating the labels for the annealed sampling level
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    # iterations for noise level. 
                    noise = torch.randn_like(x_mod,
                            device=x_mod.device) * \
                            np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise

            return torch.clamp(x_mod, 0.0, 1.0).to('cpu')

    def anneal_Langevin_dynamics_conditional(self, x_mod, scorenet, sigmas, \
            n_steps_each=100, step_lr=0.00002, stride=20, starting_index=2):
        """
        This method is a annealed langevin dynamics solver where
        the starting point is not random noise, but an image. 
        The solver is not run from the max to min noise level,
        but rather from a starting_index to the minimum noise
        level. 
        """
        model_size = self.config.data.image_size
        sigmas = sigmas[starting_index:]

        print(f'n_steps_each: {n_steps_each}.')

        # computing the domain times. 
        x_domain_times, y_domain_times = \
                (torch.tensor(x_mod.shape[2:]) - (model_size - stride)) // stride

        # perterb the input based on the noise level it should be at. 
        x_mod = x_mod + torch.randn_like(x_mod) * sigmas[0]

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), \
                    desc='annealed Langevin dynamics sampling'):
                # adjusting c:
                c = c + starting_index

                # creating the labels for the annealed sampling level
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    # iterations for noise level. 
                    for x_shift in range(x_domain_times):
                        for y_shift in range(y_domain_times):
                            # convolutional update
                            noise = torch.randn(x_mod.shape[0], 1, model_size, model_size, \
                                    device=x_mod.device) * \
                                    np.sqrt(step_size * 2)
                            grad = scorenet(x_mod[:, :, x_shift*stride:x_shift*stride+model_size, \
                                                        y_shift*stride:y_shift*stride+model_size], \
                                                        labels)
                            x_mod[:, :, x_shift*stride:x_shift*stride+model_size, \
                                    y_shift*stride:y_shift*stride+model_size] = \
                                    x_mod[:, :, x_shift*stride:x_shift*stride+model_size, \
                                    y_shift*stride:y_shift*stride+model_size] + \
                                    step_size * grad + noise

            return torch.clamp(x_mod, 0.0, 1.0).to('cpu')

    def anneal_Langevin_dynamics_large_image(self, x_mod, scorenet, sigmas, \
            n_steps_each=100, step_lr=0.00002, stride=20, domain_times=20):

        model_size = self.config.data.image_size

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), \
                    desc='annealed Langevin dynamics sampling'):
                # creating the labels for the annealed sampling level
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    # iterations for noise level. 
                    for x_shift in range(domain_times):
                        for y_shift in range(domain_times):
                            # convolutional update
                            noise = torch.randn(x_mod.shape[0], 1, model_size, model_size, \
                                    device=x_mod.device) * \
                                    np.sqrt(step_size * 2)
                            grad = scorenet(x_mod[:, :, x_shift*stride:x_shift*stride+model_size, \
                                                        y_shift*stride:y_shift*stride+model_size], \
                                                        labels)
                            x_mod[:, :, x_shift*stride:x_shift*stride+model_size, \
                                    y_shift*stride:y_shift*stride+model_size] = \
                                    x_mod[:, :, x_shift*stride:x_shift*stride+model_size, \
                                    y_shift*stride:y_shift*stride+model_size] + \
                                    step_size * grad + noise

            return torch.clamp(x_mod, 0.0, 1.0).to('cpu')


    def test_simple(self, save_name='test.pth', num_samples=50, \
                          model_name='default', binary_flag=False):
        """
        This method is a copy of the base 'test' method (from above). However,
        I have changed it to only save the final piece of information. No need
        to save every freaking thing. 
        """
        if self.config.optim.swa.lower() == 'none':
            if model_name.lower() == 'default':
                model_name = 'checkpoint.pth'

            states = torch.load(os.path.join(self.args.log, model_name), map_location=self.config.device)
            # initialization the model
            deep_model = self.get_model()
            score = deep_model(self.config).to(self.config.device)
            #score = CondRefineNetDilated(self.config).to(self.config.device)
            score = torch.nn.DataParallel(score)

            score.load_state_dict(states[0])
        else:
            if model_name.lower() == 'default':
                model_name = 'swa_checkpoint.pth'
            elif not model_name.__contains__('swa_'):
                # i wouldn't fully trust this. 
                model_name = 'swa_' + model_name

            states = torch.load(os.path.join(self.args.log, model_name), map_location=self.config.device)
            # initialization the model
            deep_model = self.get_model()
            score = deep_model(self.config).to(self.config.device)
            #score = CondRefineNetDilated(self.config).to(self.config.device)
            score = torch.nn.DataParallel(score)
            score = self.get_swa_model(score)
            score.load_state_dict(states)


        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()

        imgs = []
        if self.config.data.dataset == 'NBSA':
            # making some changes here. 
            #samples = torch.rand(num_samples, 1, self.config.data.image_size, \
            #        self.config.data.image_size, device=self.config.device)
            samples = sigmas[0] * torch.randn(num_samples, self.config.data.channels, self.config.data.image_size, \
                    self.config.data.image_size, device=self.config.device)
            all_samples = self.anneal_Langevin_dynamics_NOCLAMP(samples, score, sigmas, 1, 0.00002)[-1]

            print('------------------')
            print('testing')
            print(all_samples.shape)
            print(f"Minimum: {all_samples.min()}.")
            print(f"Maximum: {all_samples.max()}.")
            print('------------------')

            if binary_flag:
                assert all_samples.shape[1] == 1, "Only two phase samples are currently supported."
                all_samples = binarize(all_samples)


            if type(save_name) is str:
                torch.save(all_samples, os.path.join(self.args.image_folder, save_name))
            elif save_name is None:
                return all_samples
            else:
                raise NotImplementedError(f"save_name={save_name} is not accepted.")
        else:
            raise NotImplementedError("Only dataset='NBSA' is supported.")

