import torch
from torch import nn
import torch.nn.functional as F
import math
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np
import random
from PIL import Image
import torchvision.utils as vutils

"""
Some content adapted from the following:
@article{fang2019datafree,
    title={Data-Free Adversarial Distillation},	
    author={Gongfan Fang and Jie Song and Chengchao Shen and Xinchao Wang and Da Chen and Mingli Song},	  
    journal={arXiv preprint arXiv:1912.11006},	
    year={2019}
}
@inproceedings{yin2020dreaming,
	title = {Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion},
	author = {Yin, Hongxu and Molchanov, Pavlo and Alvarez, Jose M. and Li, Zhizhong and Mallya, Arun and Hoiem, Derek and Jha, Niraj K and Kautz, Jan},
	booktitle = {The IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)},
	month = June,
	year = {2020}
}
"""

class Teacher(nn.Module):

    def __init__(self, solver, generator, gen_opt, img_shape, iters, class_idx, deep_inv_params, train = True, config=None):

        super().__init__()
        self.solver = solver
        self.gen_opt = gen_opt
        self.solver.eval()
        self.generator.eval()
        self.img_shape = img_shape
        self.iters = iters
        self.config = config

        # hyperparameters
        self.di_lr = deep_inv_params[0]
        self.r_feature_weight = deep_inv_params[1]
        self.di_var_scale = deep_inv_params[2]
        self.content_temp = deep_inv_params[3]
        self.content_weight = deep_inv_params[4]

        self.image_resolution = 224
        self.random_label = False
        self.start_noise = True
        self.detach_student = False
        self.do_flip = True
        self.store_best_images = False
        self.use_fp16=True
        self.jitter=30
        self.setting_id=0
        

        # get class keys
        self.class_idx = list(class_idx)
        self.num_k = len(self.class_idx)

        # first time?
        self.first_time = train

        # set up criteria for optimization
        self.criterion = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss(reduction="none").cuda()
        self.smoothing = Gaussiansmoothing(3,5,1)

        # Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in self.solver.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module, 0, self.r_feature_weight))
        self.loss_r_feature_layers = loss_r_feature_layers


    def sample(self, size, device, return_scores=False):
        
        # make sure solver is eval mode
        self.solver.eval()

        # train if first time
        if self.first_time:
            self.first_time = False
            self.get_images(bs=size, epochs=self.iters, idx=-1)

        # sample images
        with torch.no_grad():
            x_i = self.generator.sample(size)

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x_i)
        y_hat = y_hat[:, self.class_idx]

        # get predicted class-labels (indexed according to each class' position in [self.class_idx]!)
        _, y = torch.max(y_hat, dim=1)

        return (x_i, y, y_hat) if return_scores else (x_i, y)

    def generate_scores(self, x, allowed_predictions=None, return_label=False):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x)
        y_hat = y_hat[:, allowed_predictions]

        # get predicted class-labels (indexed according to each class' position in [allowed_predictions]!)
        _, y = torch.max(y_hat, dim=1)

        return (y, y_hat) if return_label else y_hat


    def generate_scores_pen(self, x):

        # make sure solver is eval mode
        self.solver.eval()

        # get predicted logit-scores
        with torch.no_grad():
            y_hat = self.solver.forward(x=x, pen=True)

        return y_hat

    def lr_policy(self, lr_fn):
        def _alr(optimizer, iteration, epoch):
            lr = lr_fn(iteration, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return _alr

    def lr_cosine_policy(self, base_lr, warmup_length, epochs):
        def _lr_fn(iteration, epoch):
            if epoch < warmup_length:
                lr = base_lr * (epoch + 1) / warmup_length
            else:
                e = epoch - warmup_length
                es = epochs - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            return lr

        return self.lr_policy(_lr_fn)

    def denormalize(self, image_tensor, use_fp16=False):
        '''
        convert floats back to input
        '''
        if use_fp16:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

        for c in range(3):
            m, s = mean[c], std[c]
            image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

        return image_tensor

    def get_image_prior_losses(self, inputs_jit):
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
                diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
        loss_var_l1 = loss_var_l1 * 255.0
        return loss_var_l1, loss_var_l2

    def clip(self, image_tensor, use_fp16=False):
        '''
        adjust the input based on mean and variance
        '''
        if use_fp16:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
        else:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        for c in range(3):
            m, s = mean[c], std[c]
            image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
        return image_tensor


    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            place_to_store = '{}/img_s{:03d}_{:05d}_id{:03d}_gpu_{}_2.jpg'.format(self.final_data_path, class_id,
                                                                                          self.num_generations, id,
                                                                                          local_rank)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)



    def get_images(self, bs=256, epochs=1000, idx=-1):

        # clear cuda cache
        torch.cuda.empty_cache()
        #
        # self.generator.train()

        net_teacher = self.net_teacher
        use_fp16 = self.use_fp16
        save_every = self.save_every
        kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion
        img_original = 224
        data_type = torch.half if use_fp16 else torch.float
        inputs = torch.randn((bs, 3, img_original, img_original), requires_grad=True, device='cuda', dtype=data_type)
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)


        if self.setting_id==0:
            skipfirst = False
        else:
            skipfirst = True

        iteration = 0
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it == 0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000
                if self.setting_id == 2:
                    iterations_per_layer = 20000

            if lr_it == 0 and skipfirst:
                continue
            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

            if self.setting_id == 0:
                #multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True

            if use_fp16:
                static_loss_scale = 256
                static_loss_scale = "dynamic"
                _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)

            lr_scheduler = self.lr_cosine_policy(self.lr, 100, iterations_per_layer)

            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                else:
                    inputs_jit = inputs

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                net_teacher.zero_grad()

                outputs = net_teacher(inputs_jit)
                outputs = self.network_output_function(outputs)

                # R_cross classification loss
                loss = criterion(outputs, torch.argmax(outputs, dim=1))
                # R_prior losses
                loss_var_l1, loss_var_l2 = self.get_image_prior_losses(inputs_jit)

                # R_feature loss
                rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers)-1)]
                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])

                # R_ADI
                loss_verifier_cig = torch.zeros(1)
                # if self.adi_scale != 0.0:
                #     if self.detach_student:
                #         outputs_student = net_student(inputs_jit).detach()
                #     else:
                #         outputs_student = net_student(inputs_jit)
                #
                #     T = 3.0
                #     if 1:
                #         T = 3.0
                #         # Jensen Shanon divergence:
                #         # another way to force KL between negative probabilities
                #         P = nn.functional.softmax(outputs_student / T, dim=1)
                #         Q = nn.functional.softmax(outputs / T, dim=1)
                #         M = 0.5 * (P + Q)
                #
                #         P = torch.clamp(P, 0.01, 0.99)
                #         Q = torch.clamp(Q, 0.01, 0.99)
                #         M = torch.clamp(M, 0.01, 0.99)
                #         eps = 0.0
                #         loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                #         # JS criteria - 0 means full correlation, 1 - means completely different
                #         loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
                #
                #     if local_rank == 0:
                #         if iteration % save_every == 0:
                #             print('loss_verifier_cig', loss_verifier_cig.item())

                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                # combining losses
                loss_aux = self.var_scale_l2 * loss_var_l2 + \
                           self.var_scale_l1 * loss_var_l1 + \
                           self.bn_reg_scale * loss_r_feature + \
                           self.l2_scale * loss_l2

                if self.adi_scale!=0.0:
                    loss_aux += self.adi_scale * loss_verifier_cig

                loss = self.main_loss_multiplier * loss + loss_aux

                if local_rank==0:
                    if iteration % save_every==0:
                        print("------------iteration {}----------".format(iteration))
                        print("total loss", loss.item())
                        print("loss_r_feature", loss_r_feature.item())
                        print("main criterion", criterion(outputs, torch.argmax(outputs, dim=1)).item()) # TODO

                        if self.hook_for_display is not None:
                            self.hook_for_display(inputs, torch.argmax(outputs, dim=1)) # TODO

                # do image update
                if use_fp16:
                    # optimizer.backward(loss)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

                # clip color outlayers
                if do_clip:
                    inputs.data = self.clip(inputs.data, use_fp16=use_fp16)

                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

                if iteration % save_every == 0 and (save_every > 0):
                    if local_rank == 0:
                        vutils.save_image(inputs,
                                          '{}/best_images/output_{:05d}_gpu_{}.png'.format(self.prefix,
                                                                                           iteration // save_every,
                                                                                           local_rank),
                                          normalize=True, scale_each=True, nrow=int(10))

            if self.store_best_images:
                best_inputs = self.denormalize(best_inputs)
                self.save_images(best_inputs, torch.argmax(outputs, dim=1)) #TODO


        # for epoch in range(epochs):

            # sample from generator
            # inputs = self.generator.sample(bs)

            # forward with images
            # self.gen_opt.zero_grad()
            # self.solver.zero_grad()
            #
            # # content
            # outputs = self.solver(inputs)[:,:self.num_k]
            # loss = self.criterion(outputs / self.content_temp, torch.argmax(outputs, dim=1)) * self.content_weight
            #
            # # class balance
            # softmax_o_T = F.softmax(outputs, dim = 1).mean(dim = 0)
            # loss += (1.0 + (softmax_o_T * torch.log(softmax_o_T) / math.log(self.num_k)).sum())
            #
            # # R_feature loss
            # for mod in self.loss_r_feature_layers:
            #     loss_distr = mod.r_feature * self.r_feature_weight / len(self.loss_r_feature_layers)
            #     if len(self.config['gpuid']) > 1:
            #         loss_distr = loss_distr.to(device=torch.device('cuda:'+str(self.config['gpuid'][0])))
            #     loss = loss + loss_distr
            #
            # # image prior
            # inputs_smooth = self.smoothing(F.pad(inputs, (2, 2, 2, 2), mode='reflect'))
            # loss_var = self.mse_loss(inputs, inputs_smooth).mean()
            # loss = loss + self.di_var_scale * loss_var
            #
            # # backward pass
            # loss.backward()
            #
            # self.gen_opt.step()

        # clear cuda cache
        torch.cuda.empty_cache()

class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, gram_matrix_weight, layer_weight):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.target = None
        self.gram_matrix_weight = gram_matrix_weight
        self.layer_weight = layer_weight

    def hook_fn(self, module, input, output):

        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        # var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False) + 1e-8
        # r_feature = torch.log(var**(0.5) / (module.running_var.data.type(var.type()) + 1e-8)**(0.5)).mean() - 0.5 * (1.0 - (module.running_var.data.type(var.type()) + 1e-8 + (module.running_mean.data.type(var.type())-mean)**2)/var).mean()
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)

        self.r_feature = r_feature

            
    def close(self):
        self.hook.remove()

class Gaussiansmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(Gaussiansmoothing, self).__init__()
        kernel_size = [kernel_size] * dim
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).cuda()

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)
