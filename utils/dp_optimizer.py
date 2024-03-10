import numpy as np
import torch
from torch.optim import SGD, Adam, Adagrad, RMSprop


def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):

            super().__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            for id,group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
        def zero_microbatch_grad(self):
            super().zero_grad()


        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2

            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm


        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()


        def step_dp(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super().step(*args, **kwargs)

        def compute_L2norm_perlayer(self):
            num_perplayer = len(self.param_groups)
            perlayer_norm = np.zeros(num_perplayer, dtype=np.float64, order='C')
            for i, group in enumerate(self.param_groups):
                layer_norm = 0.0
                for param in group['params']:
                    if param.requires_grad:
                        layer_norm += param.grad.data.norm(2).item() ** 2
                perlayer_norm[i] = layer_norm ** 0.5

            return perlayer_norm
        
        def microbatch_step_perlayer(self):
            num_perlayer = len(self.param_groups)
            perlayer_norm = self.compute_L2norm_perlayer()
            clip_coef = np.zeros(num_perlayer, dtype=np.float64, order='C')
            l2_norm_clip_perlayer = []
            total_l2norm_square = np.sum(np.square(perlayer_norm))
            for norm in perlayer_norm:
                proposition = (norm ** 2) / total_l2norm_square
                clip_threshold = proposition ** 0.5 * self.l2_norm_clip
                l2_norm_clip_perlayer.append(clip_threshold)

            for i in range(num_perlayer):
                clip_coef[i] = min(l2_norm_clip_perlayer[i] / (perlayer_norm[i] + 1e-6), 1.)

            for i, group in enumerate(self.param_groups):
                for param in group['params']:
                    if param.requires_grad:
                        param.grad.data.mul_(clip_coef[i])

            return perlayer_norm
        
        def step_dp_perlayer(self, *args, **kwargs):
            perlayer_norm = self.compute_L2norm_perlayer()
            total_l2norm_square = np.sum(np.square(perlayer_norm))
            l2_norm_clip_perlayer = []

            for norm in perlayer_norm:
                proposition = (norm ** 2) / total_l2norm_square
                clip_threshold = proposition ** 0.5 * self.l2_norm_clip
                l2_norm_clip_perlayer.append(clip_threshold)

            for i, group in enumerate(self.param_groups):
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(l2_norm_clip_perlayer[i] * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            
            super().step(*args, **kwargs)


        def microbatch_step_auto(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2

            total_norm = total_norm ** 0.5
            clip_coef = 1 / (total_norm + 1e-2)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm


        def step_dp_auto(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super().step(*args, **kwargs)


    return DPOptimizerClass

DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)

def get_dp_optimizer(lr, momentum, C_t, sigma, batch_size, model):
    optimizer = DPAdam_Optimizer(
        l2_norm_clip=C_t,
        noise_multiplier=sigma,
        minibatch_size=batch_size,
        microbatch_size=1,
        params=model.parameters(),
        lr=lr,
        momentum=momentum
    )

    return optimizer

