import torch
from torch import nn

from cryptorch.utils import fetch_attr, set_attr

# Wrap around your net with this wrapper to export the backward graph
class TrainWrapper(nn.Module):
    def __init__(self, net, loss_func):
        super().__init__()
        self.base_model = net
        self.loss_func = loss_func

    def forward(self, x, y):
        pred = self.base_model(x)
        loss = self.loss_func(pred, y)
        return loss

class CrypTorchSGDOptimizer:
    def __init__(self, module, graph_signature, lr, param_map={}):
        self.module = module
        self.graph_signature = graph_signature
        self.lr = lr
        self.param_map = param_map

    def step(self, grads):
        loss = None
        for spec, grad in zip(self.graph_signature.output_specs, grads):
            if spec.target is None:
                # There should be only one loss!
                assert(loss is None)
                loss = grad
            else:
                if len(self.param_map) == 0:
                    target = spec.target
                else:
                    target = self.param_map[spec.target]

                t = fetch_attr(f"{target}.data", self.module)

                if t.dtype == torch.int64:
                    # TODO: encoding scale is hardcoded here.
                    new_t = t - torch.ops.cryptorch.div(grad * int(self.lr * 2 ** 16), 2 ** 16)
                else:
                    new_t = t - grad * self.lr

                set_attr(f"{target}.data", new_t, self.module)
        return loss

# nn.CrossEntropyLoss internally has ignore_index,
# which complicates the computation.
# Thus, we use a simplified custom loss function instead.
# Also, following CrypTen, we assume the target is in one-hot vector.
def simplified_cross_entropy_loss(pred, y):
    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    target_log_probs = log_probs * y
    return -target_log_probs.sum() / pred.size(0)

def load_trained_weights_into(ep, trained_module, param_map={}):
    mod_new = ep.module()
    for spec in ep.graph_signature.input_specs:
        if spec.target is not None:
            if len(param_map) == 0:
                target = f"base_model.{spec.target}"
            else:
                target = param_map[f"base_model.{spec.target}"]
            t = fetch_attr(f"{target}.data", trained_module)
            if len(param_map) > 0:
                t = torch.ops.cryptorch.decrypt(t, 16)
            set_attr(f"{spec.target}.data", t, mod_new)

    return mod_new
