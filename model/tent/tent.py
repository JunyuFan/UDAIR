from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import torch.nn.functional as F
from .coral import CORAL


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)


    def forward(self, x, source_domain_distrib, reference_index):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            results = forward_and_adapt(x, source_domain_distrib, self.model, self.optimizer, reference_index)

        return results

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)




@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, source_domain_distrib, model, optimizer, reference_index):
    """Forward once, align degradation distributions, and update adapter params."""
    results = model(x)

    if type(results) is not torch.Tensor:
        target_dist_4 = results[3]
        target_dist_3 = results[4]
        target_dist_2 = results[5]
    else:
        return results

    source_dist_4 = source_domain_distrib[0]
    source_dist_3 = source_domain_distrib[1]
    source_dist_2 = source_domain_distrib[2]

    if target_dist_4.shape[-1] != source_dist_4.shape[-1]:
        target_dist_4 = F.adaptive_avg_pool2d(target_dist_4, source_dist_4.shape[-2:])
    if target_dist_3.shape[-1] != source_dist_3.shape[-1]:
        target_dist_3 = F.adaptive_avg_pool2d(target_dist_3, source_dist_3.shape[-2:])
    if target_dist_2.shape[-1] != source_dist_2.shape[-1]:
        target_dist_2 = F.adaptive_avg_pool2d(target_dist_2, source_dist_2.shape[-2:])

    def prepare_feat(feat, is_source=False):
        """Convert feature maps to (samples, channels)."""
        if is_source:
            feat = feat[reference_index]
            if feat.dim() == 3:
                return feat.permute(1, 2, 0).reshape(-1, feat.size(0))
            if feat.dim() == 4:
                return feat.permute(0, 2, 3, 1).reshape(-1, feat.size(1))
            raise ValueError(f"Unsupported source feature shape: {tuple(feat.shape)}")
        return feat.permute(0, 2, 3, 1).reshape(-1, feat.size(1))

    s_feat_4 = prepare_feat(source_dist_4, is_source=True)
    s_feat_3 = prepare_feat(source_dist_3, is_source=True)
    s_feat_2 = prepare_feat(source_dist_2, is_source=True)

    t_feat_4 = prepare_feat(target_dist_4)
    t_feat_3 = prepare_feat(target_dist_3)
    t_feat_2 = prepare_feat(target_dist_2)

    loss1 = CORAL(s_feat_4, t_feat_4)
    loss2 = CORAL(s_feat_3, t_feat_3)
    loss3 = CORAL(s_feat_2, t_feat_2)
    loss = (loss1 + loss2 + loss3) * 1e5

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return results

def collect_params(model, modules_identifier=""):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if modules_identifier in nm:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, modules_identifier=""):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for name, m in model.named_modules():
        if modules_identifier in name:
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


