import torch, math
from torch.optim.optimizer import Optimizer

class AvaGrad(Optimizer):
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), weight_decay=0, eps=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, gamma=None)
        super(AvaGrad, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AvaGrad, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            gamma = group['gamma']
            squared_norm = 0.0
            num_params = 0.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                prev_bias_correction2 = 1 - beta2 ** (state['step']-1)
                
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                sqrt_exp_avg_sq = exp_avg_sq.sqrt()

                if state['step'] > 1:
                    denom = sqrt_exp_avg_sq.div(math.sqrt(prev_bias_correction2)).add_(eps)
                    step_size = gamma * group['lr'] / bias_correction1
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                param_wise_lr = sqrt_exp_avg_sq.div_(math.sqrt(bias_correction2)).add_(eps)
                squared_norm += param_wise_lr.norm(-2)**(-2)
                num_params += param_wise_lr.numel()

            group['gamma'] = 1. / (squared_norm / num_params).sqrt()
        return loss


class AvaGradW(Optimizer):
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), weight_decay=0, eps=0.1):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, gamma=None)
        super(AvaGradW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AvaGradW, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            gamma = group['gamma']
            squared_norm = 0.0
            num_params = 0.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                eps = group['eps']
                
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                prev_bias_correction2 = 1 - beta2 ** (state['step']-1)
                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                sqrt_exp_avg_sq = exp_avg_sq.sqrt()

                if state['step'] > 1:
                    denom = sqrt_exp_avg_sq.div(math.sqrt(prev_bias_correction2)).add_(eps)
                    step_size = gamma * group['lr'] / bias_correction1
                    
                    if group['weight_decay'] != 0:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])                     
                    
                    p.data.addcdiv_(-step_size, exp_avg, denom)

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                param_wise_lr = sqrt_exp_avg_sq.div_(math.sqrt(bias_correction2)).add_(eps)
                squared_norm += param_wise_lr.norm(-2)**(-2)
                num_params += param_wise_lr.numel()

            group['gamma'] = 1. / (squared_norm / num_params).sqrt()
        return loss
