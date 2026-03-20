import torch
from torch import nn
import math
from typing import Optional, Iterable, Dict, Any
from cryptorch.utils import is_secret
    
def get_passes_from_configs(configs: Iterable[Dict[str, Any]]):
    return [get_pass_by_name(**config) for config in configs]   

def get_pass_by_name(name: str, **kwags):
    import sys
    mod = sys.modules[__name__]
    
    if hasattr(mod, name):
        return getattr(mod, name)(**kwags)
    else:
        raise ValueError(f"Unknown pass: {name}")

class BasePass:
    name = None
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __str__(self):
        return self.get_name()
    
    def get_name(self):
        if self.name is not None:
            return self.name
        else:
            return self.__class__.__name__
    
    def __repr__(self):
        arg_strings = map(lambda item: f"{item[0]}={item[1]!r}", self.kwargs.items())
        return f"{self.get_name()}({', '.join(arg_strings)})"
    
    def get_match_filters(self):
        return []
    def get_patterns(self):
        raise NotImplementedError()
    def get_replacement(self):
        raise NotImplementedError()

class TunablePass(BasePass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_replacement(self, knobs):
        raise NotImplementedError()

    def get_max_knob(self):
        return tuple(map(lambda x: x[-1], self.get_possible_knob_positions()))

    def decrement_knob(self, knob, i):
        current_knob = knob[i]
        possible_knobs = self.get_possible_knob_positions()[i]
        
        if current_knob not in possible_knobs:
            raise RuntimeError(f"current knob state {current_knob} (index {i}) is not valid. Valid values are {', '.join(map(str, possible_knobs))}")
        
        current_knob_index = possible_knobs.index(current_knob)
        
        if current_knob_index == 0:
            # knob already at min value
            return None
        else:
            new_knob = possible_knobs[current_knob - 1]
            new_knobs = list(knob)
            new_knobs[i] = new_knob
            return tuple(new_knob)
            
        

    def increment_knob(self, knob, i):
        current_knob = knob[i]
        possible_knobs = self.get_possible_knob_positions()[i]
        
        if current_knob not in possible_knobs:
            raise RuntimeError(f"current knob state {current_knob} (index {i}) is not valid. Valid values are {', '.join(map(str, possible_knobs))}")
        
        current_knob_index = possible_knobs.index(current_knob)
        
        if current_knob_index == (len(possible_knobs - 1)):
            # knob already at max value
            return None
        else:
            new_knob = possible_knobs[current_knob + 1]
            new_knobs = list(knob)
            new_knobs[i] = new_knob
            return tuple(new_knob)

    def get_min_knob(self):
        return tuple(map(lambda x: x[0], self.get_possible_knob_positions()))
    
    def get_possible_knob_positions(self):
        raise NotImplementedError
    

class WherePass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda cond, x, y: torch.ops.aten.where(cond, x, y),
                lambda cond, x, y: torch.ops.aten.where.default(cond, x, y),
                lambda cond, x, y: torch.ops.aten.where.self(cond, x, y),
                lambda cond, x, y: torch.ops.aten.where.ScalarOther(cond, x, y)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, cond, x, y):
            #return cond * x + (~cond) * y
            return y + (x - y) * cond

'''
class MaskedFillRewritePass(BasePass):
    def __init__(self):
        super().__init__("MaskedFillRewritePass")

    def get_patterns(self):
        return [lambda x, mask, val: torch.ops.aten.masked_fill.Scalar(x, mask, val)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def __init__(self, minval):
            super().__init__()
            self.minval = minval

        def forward(self, x, mask, minval):
            return x.masked_fill(mask, self.minval)
'''

class ReluPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x: torch.ops.aten.relu_.default(x),
                lambda x: torch.ops.aten.relu.default(x)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x):
            return x * (x >= 0).float()

class AddmmPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda input, mat1, mat2: torch.ops.aten.addmm.default(input, mat1, mat2)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, input, mat1, mat2):
            return torch.mm(mat1, mat2) + input

class GePass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x, y: torch.ops.aten.ge.Scalar(x, y),
                lambda x, y: torch.ops.aten.ge.Tensor(x, y)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x, y):
            return ~(x - y < 0)
            #return ~torch.ops.cryptorch.ltz(x - y)

class GtPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x, y: torch.ops.aten.gt.Scalar(x, y),
                lambda x, y: torch.ops.aten.gt.Tensor(x, y)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x, y):
            return (y - x < 0)

class LePass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x, y: torch.ops.aten.le.Scalar(x, y),
                lambda x, y: torch.ops.aten.le.Tensor(x, y)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x, y):
            return ~(y - x < 0)
            #return ~torch.ops.cryptorch.ltz(y - x)

class LtPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x, y: torch.ops.aten.lt.Scalar(x, y),
                lambda x, y: torch.ops.aten.lt.Tensor(x, y)]

    def get_replacement(self):
        return self.Replacement()

    def get_match_filters(self):
        def f(head):
            return head.args[1] != 0
        return [f]

    class Replacement(nn.Module):
        def forward(self, x, y):
            return x - y < 0
            #return torch.ops.cryptorch.ltz(x - y)


class PolyApprox(nn.Module):
    def __init__(self, coefficents):
       super().__init__()
       self.coefficents = coefficents

    def forward(self, x):
        approx = self.coefficents[1] * x + self.coefficents[0]
        power_of_x = x
        for coefficent in self.coefficents[2:]:
            power_of_x = power_of_x * x
            approx = approx + coefficent * power_of_x
        return approx

# Produce spcified orders of the power of x or optionally abs(x)
def generate_powers_of_x(x: torch.Tensor, order: int, take_abs: bool = False, sign_x: Optional[torch.Tensor] = None):
    if order < 1:
        raise ValueError(f"Invalid order {order}, must be at least 1")
    
    if take_abs and sign_x is None:
        sign_x = torch.sign(x)
    
    if order == 1:
        return torch.stack([sign_x * x if take_abs else x,])
    
    num_steps = (order - 1).bit_length()
    
    # hardcoded step 1, fused with abs
    if take_abs:
        result = torch.stack([x, x]) * torch.stack([sign_x, x])
    else:
        result = torch.stack([x, x * x])
    
    for _ in range(num_steps - 1):
        current_orders = result.size(0)
        remining_orders = order - current_orders
        width = min(remining_orders, current_orders)
        
        print(width)
        
        new_part = result[:width] * result[-1]
        result = torch.cat([result, new_part])
        
    return result
    
class PolyApprox2(nn.Module):
    def __init__(self, coefficents, take_abs=False, pre_scale=1.0):
        super().__init__()
    #    self.bias = coefficents[0]
        # self.register_buffer("coefficents", torch.tensor(coefficents))
        self.coefficents = coefficents
        self.take_abs=take_abs
        self.pre_scale = pre_scale

    def forward(self, x: torch.Tensor, sign_x: torch.Tensor=None, return_x=False):
        powers = generate_powers_of_x(x * self.pre_scale, len(self.coefficents) - 1, self.take_abs, sign_x)
        # result = torch.matmul(torch.tensor(self.coefficents[1:]), powers) + self.coefficents[0]
        
        post_scale = self.pre_scale
        result = powers[0] * (self.coefficents[1] / post_scale) + self.coefficents[0]
        for c, power in zip(self.coefficents[2:], powers[1:]):
            post_scale = post_scale * self.pre_scale
            result = result + power * (c / (post_scale))
            
        
        if return_x:
            return result, powers[0] * (1 / self.pre_scale)
        else:
            return result
        

class SigmoidPass(BasePass):
    def __init__(self, approx):
        super().__init__(approx=approx)
        self.approx = approx

    def get_patterns(self):
        return [lambda x: torch.ops.aten.sigmoid.default(x)]

    def get_replacement(self):
        if self.approx == "crypten-reciprocal":
            return self.CryptenReciprocal()
        elif self.approx == "bolt":
            return self.Bolt()
        elif self.approx == "bolt7":
            return self.Bolt7()
        elif self.approx == "rsqrt":
            return self.RSqrt()
        elif self.approx == "tanh":
            return self.Tanh()
        else:
            raise NotImplementedError()

    class CryptenReciprocal(nn.Module):
        def forward(self, x):
            sign = x.sign()
            pos_x = sign * x
            out = 1 / ((-pos_x).exp() + 1)
            return out * ((sign + 1) / 2) + (1 - out) * (-(sign - 1) / 2)
    
    class Bolt(nn.Module):
        def __init__(self):
            super().__init__()
            self.poly = PolyApprox(coefficents=(0.0026799414038387176, 0.29509062970411726, -0.06539687820531254, 0.006338689352443927, -0.00022553131096302256))

        def forward(self, x):
            
            sign_x = x.sign()
            pos_x = x * sign_x
            cond = pos_x < 5
            # x_is_pos = x > 0

            approx = self.poly(pos_x)

            # approx = torch.where(cond, approx, 0.5)
            approx = cond * approx + (1 - cond) * 0.5
            approx = approx * sign_x + 0.5
            

            return approx

    class Bolt7(nn.Module):
        def __init__(self):
            super().__init__()
            self.poly = PolyApprox(coefficents=(-0.0007794947065195241, 0.25684992354248903, -0.007041942014251339, -0.024793997667995903, 0.007707003922680195, -0.001045807692238169, 6.88262552051938e-05, -1.788587797205144e-06))

        def forward(self, x):
            
            sign_x = x.sign()
            pos_x = x * sign_x
            cond = pos_x < 5
            # x_is_pos = x > 0

            approx = self.poly(pos_x)

            approx = torch.where(cond, approx, 0.5)
            approx = approx * sign_x + 0.5
            

            return approx

    class RSqrt(nn.Module):
        
        # https://en.wikipedia.org/wiki/Sigmoid_function#Examples
        # 0.5 * x * rsqrt(x^2 + 1) + 0.5
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * 0.6 * torch.rsqrt(x * x * 0.36 + 1) + 0.5
    
    class Tanh(nn.Module):
        
        # https://en.wikipedia.org/wiki/Sigmoid_function#Examples
        # 0.5 * x * rsqrt(x^2 + 1) + 0.5
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 + 0.5 * torch.tanh(0.5 * x)
        
class TunableSigmoidPass(TunablePass):
    def __init__(self):
        super().__init__()
        self.max_order = 4
        self.min_order = 0

    def get_patterns(self):
        return [lambda x: torch.ops.aten.sigmoid.default(x)]


    def get_replacement(self, knobs):
        order = knobs[0]
        if order == 4: 
            return self.Bolt_O4()
        elif order == 2:
            return self.Bolt_O2()
        elif order == 0:
            return self.Bolt_O0()
        else:
            raise NotImplementedError()
    
    class Bolt_O4(nn.Module):
        def forward(self, x):
            # g0 = 0.6012961737280229
            # g1 = -0.4818827760495077
            # g2 = 1.0029772762860651
            # g3 = -3.6068525972536896
            # g4 = 2.609291300562604
            # limit = 5
            
            g0 = 0.568008926081178
            g1 = -0.32150403697986685
            g2 = 1.4239765285787307
            g3 = -4.913616906648125
            g4 = 4.966474201573112
            limit = 5.3
            
            sign_x = x.sign()
            pos_x = x * sign_x
            cond = pos_x <= limit
            scaled_pos_x = pos_x * (1 / limit)
            
            sigmoid_p0 = (g0 * scaled_pos_x + g1) * scaled_pos_x + g2
            sigmoid_p1 = (sigmoid_p0 + g0 * scaled_pos_x + g3) * sigmoid_p0 + g4

            return torch.where(cond, sigmoid_p1 * sign_x + 0.5, (sign_x + 1) / 2)
    
    class Bolt_O2(nn.Module):
        def forward(self, x):
            a0 = 0.010689877129844667
            a1 = 1.0770212399619459
            a2 = -0.6105294292059095
            
            limit = 4.4

            sign_x = x.sign()
            pos_x = x * sign_x
            cond = pos_x <= limit
            scaled_pos_x = pos_x * (1 / limit)
            
            sigmoid_p0 = scaled_pos_x * scaled_pos_x * a2 + scaled_pos_x * a1 + a0 

            return torch.where(cond, sigmoid_p0 * sign_x + 0.5, (sign_x + 1) / 2)
    
    class Bolt_O0(nn.Module):
        def forward(self, x):
           return (x.sign() + 1) / 2


    # def get_max_knob(self):
    #     return (self.max_order,)

    # def decrement_knob(self, knob, i):
    #     knob_new = list(knob)
    #     if knob_new[i] > self.get_min_knob()[i]:
    #         knob_new[i] -= 2
    #         return tuple(knob_new)
    #     else:
    #         return None

    # def increment_knob(self, knob, i):
    #     knob_new = list(knob)
    #     if knob_new[i] < self.get_max_knob()[i]:
    #         knob_new[i] += 2
    #         return tuple(knob_new)
    #     else:
    #         return None

    # def get_min_knob(self):
    #     return (self.min_order,)
    
    def get_possible_knob_positions(self):
        return ((0, 2, 4, 6),)

class HardsigmoidPass(BasePass):
    def __init__(self):
        super().__init__()

    def get_patterns(self):
        return [lambda x: torch.ops.aten.hardsigmoid.default(x),
                lambda x: torch.ops.aten.hardsigmoid_.default(x)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(torch.nn.Module):
        def forward(self, x):
            cond1 = x > -3
            cond2 = x < 3
            return (x * (1 / 6) + 0.5) * cond1 * cond2 + (~cond2)


class HardswishPass(BasePass):
    def __init__(self):
        super().__init__()

    def get_patterns(self):
        return [lambda x: torch.ops.aten.hardswish.default(x),
                lambda x: torch.ops.aten.hardswish_.default(x)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hardsigmoid = torch.nn.Hardsigmoid()
        
        def forward(self, x):
            return x * self.hardsigmoid(x)
    
class TanhPass(BasePass):
    def __init__(self, approx):
        super().__init__(approx=approx)
        self.approx = approx

    def get_patterns(self):
        return [lambda x: torch.ops.aten.tanh.default(x)]

    def get_replacement(self):
        if self.approx == "bolt":
            return self.Bolt()
        elif self.approx == "bolt2" or self.approx == "poly":
            return self.Bolt2()
        elif self.approx == "crypten-reciprocal":
            return self.CryptenReciprocal()
        elif self.approx == "spu":
            return self.SPU()
        else:
            raise NotImplementedError()

    class Bolt(nn.Module):
        def forward(self, x):
            a = -0.013232131886235352
            b = 0.09948747962825866
            c = -0.20093640347818847
            d = -0.17616532856475706
            e = 1.0542492677156243
            f = -0.0024920889620412097
            sign_x = x.sign()
            
            pos_x = x * sign_x
            cond1 = pos_x > 2.855
            # cond1 = pos_x > 3
            x2 = x * x
            x3 = x2 * pos_x
            x4 = x3 * pos_x
            x5 = x4 * pos_x
            return ((a * x5 + b * x4 + c * x3 + d * x2 + e * pos_x + f) * (~cond1) + cond1) * sign_x
    
    # class Bolt2(nn.Module):
    #     def __init__(self):
    #         super().__init__()
            
    #         self.coefficents = (
    #             -0.0024920889620412097,
    #             1.0542492677156243,
    #             -0.17616532856475706,
    #             -0.20093640347818847,
    #             0.09948747962825866,
    #             -0.013232131886235352,
    #             )
        
    #     def forward(self, x):
    #         sign_x = x.sign()
    #         powers = generate_powers_of_x(x, len(self.coefficents) - 1, take_abs=True, sign_x=sign_x)
    #         pos_x = powers[0]
    #         # approx = torch.matmul(torch.movedim(powers, 0, -1), self.coefficents[1:]) + self.coefficents[0]
    #         approx = pos_x * self.coefficents[1] + self.coefficents[0]
    #         for c, power in zip(self.coefficents[2:], powers[1:]):
    #             approx += power * c
    #         cond1 = pos_x > 2.855
    #         return (approx * (~cond1) + cond1) * sign_x
        
    class Bolt2(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.poly_approx = PolyApprox2(coefficents=(
                    -0.0024920889620412097,
                    1.0542492677156243,
                    -0.17616532856475706,
                    -0.20093640347818847,
                    0.09948747962825866,
                    -0.013232131886235352,
                ),
                take_abs=True,
                # scales the input down by 1/3rd to make sure input is 0-1, helps pervent overflow.
                pre_scale=0.33
            )
        
        def forward(self, x):
            sign_x = x.sign()
            approx, pos_x = self.poly_approx(x, sign_x, return_x=True)
            cond1 = pos_x > 2.855
            return (approx * (~cond1) + cond1) * sign_x
            

    class CryptenReciprocal(nn.Module):
        def forward(self, x):
            return ((x * 2).sigmoid() * 2 - 1)

    class SPU(nn.Module):
        def forward(self, x):
            coeffs = [1.2514045938932097,   -0.3655987797163166,   0.17253141478140663,
                    -0.08943445792774211, 0.047703017901250824,  -0.025830290571688078,
                    0.014338801903468182, -0.008541730970059077, 0.0061230685785789475]
            x = torch.where(x > 5., 5., torch.where(x < -5., -5., x)) * 0.2
            # Degree 9 Chebyshev polynomial
            y = 4 * x * x - 2
            z = y - 1
            poly = [x, x * z]
            for i in range(2, 9):
                poly.append(y * poly[i - 1] - poly[i - 2])

            return sum([x0 * x1 for x0, x1 in zip(coeffs, poly)])


class SignPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x: torch.ops.aten.sign.default(x)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x):
            return 1 - 2 * (x < 0)


class AbsPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x: torch.ops.aten.abs.default(x)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x):
            return x.sign() * x


class LayerNormPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x, shape, gamma, beta, eps: torch.ops.aten.layer_norm.default(x, shape, gamma, beta, eps)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x, shape, gamma, beta, eps=1e-05):
            diff = (x - x.mean(dim=-1, keepdim=True))
            var = (diff * diff).mean(dim=-1, keepdim=True) + eps
            # TODO: rsqrt over sqrt is manually selected for now. Converting 1/sqrt to rsqrt can be another pass in the future.
            #return (diff * gamma) / var.sqrt() + beta
            return (diff * gamma) * var.rsqrt() + beta

    def get_match_filters(self):
        def f(head):
            return is_secret(head.args[0])
        return [f]


class GroupNormPass(BasePass):
    def __init__(self, pre_scale=1.0):
        super().__init__(pre_scale=pre_scale)
        self.pre_scale = pre_scale
    
    def get_patterns(self):
        return [lambda x, num_groups, weight, bias, eps: torch.ops.aten.group_norm.default(x, num_groups, weight, bias, eps)]

    def get_replacement(self):
        return self.Replacement(pre_scale=self.pre_scale)

    class Replacement(nn.Module):
        def __init__(self, pre_scale=1.0):
            super().__init__()
            self.pre_scale = pre_scale
        
        def forward(self, x, num_groups, weight=None, bias=None, eps=1e-05):
            intput_size = x.size()
            N, C, *remaining_dims = intput_size
            view = x.reshape((N, num_groups, -1))
            if self.pre_scale == 0:
                max = view.abs().amax(dim=-1, keepdim=True)
                view = view / max
            else:
                view = view / self.pre_scale
            mean = view.mean(-1, keepdim=True)
            diff = view - mean
            var = (diff * diff).mean(-1, keepdim=True)
            
            result = diff * ((var + eps).rsqrt())
            
            result = result.reshape((N, C, -1))
            
            if weight is not None:
                result = result * weight.unsqueeze(-1)
            if bias is not None:
                result = result + bias.unsqueeze(-1)
            
            result = result.reshape(intput_size)
            
            # affine_size = (-1,) + (1,) * (x.dim() - 2)
            
            return result

class GeluPass(BasePass):
    def __init__(self, approx):
        super().__init__(approx=approx)
        self.approx = approx

    def get_patterns(self):
        return [lambda x: torch.ops.aten.gelu.default(x)]

    def get_replacement(self):
        if self.approx == "bolt":
            return self.Bolt()
        elif self.approx == "bolt2" or self.approx == "poly":
            return self.Bolt2()
        elif self.approx == "bolt3":
            return self.Bolt3()
        elif self.approx == "poly2":
            return self.Poly2()
        elif self.approx == "crypten":
            return self.Crypten()
        elif self.approx == "erf":
            return self.Erf()
        else:
            raise NotImplementedError()

    class Bolt(nn.Module):
        def forward(self, x):
            a = 0.020848611754127593
            b = -0.18352506127082727
            c = 0.5410550166368381
            d = -0.03798164612714154
            e = 0.001620808531841547

            cond1 = x > 2.7
            cond2 = x >= -2.7
            pos_x = x.abs()

            x2 = x * x
            x3 = x2 * pos_x
            x4 = x3 * pos_x

            return x * cond1 + (a * x4 + b * x3 + c * x2 + d * pos_x + e + 0.5 * x) * (cond2) * (~cond1)
    
    class Bolt2(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.poly_approx = PolyApprox2(coefficents=(
                    0.001620808531841547,
                    -0.03798164612714154,
                    0.5410550166368381,
                    -0.18352506127082727,
                    0.020848611754127593,
                ),
                take_abs=True,
                # scales the input down by 1/3rd to make sure input is 0-1, helps pervent overflow.
                pre_scale=1/2.7
            )
        
        def forward(self, x):
            cond1 = x > 2.7
            cond2 = x >= -2.7
            approx = self.poly_approx(x)
            
            return x * cond1 + (approx + 0.5 * x) * (cond2) * (~cond1)
    
    class Bolt3(nn.Module):
        def forward(self, x):
            g0 = 0.14439048359960427
            g1 = -0.7077117131613893
            g2 = 4.5702822654246535
            g3 = -8.15444702051307
            g4 = 16.382265425072532

            pos_x = x.abs()
            relu_x = (pos_x + x) / 2
            cond = pos_x <= 2.7
            
            gelu_p0 = (g0 * pos_x + g1) * pos_x + g2
            gelu_p1 = (gelu_p0 + g0 * pos_x + g3) * gelu_p0 + g4 + 0.5 * x

            return torch.where(cond, gelu_p1, relu_x)
    
    class Poly2(nn.Module):
        def forward(self, x):
            scaled_x = x * 0.3333333
            scaled_x_2 = scaled_x * scaled_x
            scaled_x_4 = scaled_x_2 * scaled_x_2
            scaled_x_6 = scaled_x_4 * scaled_x_2
            
            poly_approx = (0.008457623257611086
            - 1.5000000000000022 * scaled_x
            + 3.246654227859503 * scaled_x_2
            - 3.0652781881780466 * scaled_x_4
            + 1.32622456457803 * scaled_x_6)
            
            conds = torch.stack([-x, x]) < 3
            
            cond_1 = conds[0]
            cond_2 = conds[1]
            
            return (x + (poly_approx) * cond_2) * cond_1
    class Erf(nn.Module):
        def forward(self, x):
            return 0.5 * (x * (1 + torch.erf(x / math.sqrt(2))))

    class Crypten(nn.Module):
        def forward(self, x):
            return 0.5 * x * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)).tanh())


class SiluPass(BasePass):
    def __init__(self, approx):
        super().__init__(approx=approx)
        self.approx = approx

    def get_patterns(self):
        return [
            lambda x: torch.ops.aten.silu.default(x),
            lambda x: torch.ops.aten.silu_.default(x)
        ]

    def get_replacement(self):
        if self.approx == "bolt":
            return self.Bolt()
        elif self.approx == "bolt3":
            return self.Bolt3()
        elif self.approx == "real" or self.approx == "sigmoid":
            return self.Real()
        else:
            raise NotImplementedError()
        
    class Real(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
            

    class Bolt(nn.Module):
        def forward(self, x):

            a0 = -0.011633682541851762
            a1 = 0.04172072904060847
            a2 = 0.26287081795168915
            a3 = -0.055805147372078076
            a4 = 0.005180352157862565
            a5 = -0.00017703117270809605

            cond1 = x > 10
            cond2 = x >= -10
            pos_x = x.abs()

            '''
            x2 = x * x
            x3 = x2 * pos_x
            x4 = x3 * pos_x
            x5 = x4 * pos_x

            return x * cond1 + (a5 * x5 + a4 * x4 + a3 * x3 + a2 * x2 + a1 * pos_x + a0 + 0.5 * x) * (cond2) * (~cond1)
            '''
            x1 = a1 * pos_x
            x2 = (a2/a1) * x1 * pos_x
            x3 = (a3/a2) * x2 * pos_x
            x4 = (a4/a3) * x3 * pos_x
            x5 = (a5/a4) * x4 * pos_x
            return x * cond1 + (x5 + x4 + x3 + x2 + x1 + a0 + 0.5 * x) * (cond2) * (~cond1)
    
    class Bolt3(nn.Module):
        def forward(self, x):
            g0 = 0.05351430810763807
            g1 = -0.4585514803234726
            g2 = 11.082829581109497
            g3 = -20.968045027134355
            g4 = 109.54588062791908

            pos_x = x.abs()
            relu_x = (pos_x + x) / 2
            cond = pos_x <= 6.5
            
            silu_p0 = (g0 * pos_x + g1) * pos_x + g2
            silu_p1 = (silu_p0 + g0 * pos_x + g3) * silu_p0 + g4 + 0.5 * x

            return torch.where(cond, silu_p1, relu_x)

class TunableSiluPass(TunablePass):
    def __init__(self):
        super().__init__()
        self.max_order = 4
        self.min_order = 0

    def get_patterns(self):
        return [
            lambda x: torch.ops.aten.silu.default(x),
            lambda x: torch.ops.aten.silu_.default(x)
        ]

    def get_replacement(self, knobs):
        order = knobs[0]
        if order == 4: 
            return self.Bolt_O4()
        elif order == 2:
            return self.Bolt_O2()
        elif order == 0:
            return self.Bolt_O0()
        else:
            raise NotImplementedError()
    
    class Bolt_O4(nn.Module):
        def forward(self, x):
            g0 = 2.2609795175477085
            g1 = -3.937152879526602
            g2 = 3.172725065372014
            g3 = -4.595969693299374
            g4 = 4.5052863722554
            limit = 6.5

            pos_x = x.abs()
            relu_x = (pos_x + x) / 2
            cond = pos_x <= limit
            scaled_pos_x = pos_x * (1 / limit)
            
            silu_p0 = (g0 * scaled_pos_x + g1) * scaled_pos_x + g2
            silu_p1 = (silu_p0 + g0 * scaled_pos_x + g3) * silu_p0 + g4 + 0.5 * x

            return torch.where(cond, silu_p1, relu_x)
    
    class Bolt_O2(nn.Module):
        def forward(self, x):
            a0 = -0.061824197242733751
            a1 = 1.2900018198094088
            a2 = 0.8988761597422655
            limit = 4.25

            pos_x = x.abs()
            relu_x = (pos_x + x) / 2
            cond = pos_x <= limit
            scaled_pos_x = pos_x * (1 / limit)
            
            gelu_p0 = scaled_pos_x * scaled_pos_x * a2 + scaled_pos_x * a1 + a0 + 0.5 * x

            return torch.where(cond, gelu_p0, relu_x)
    
    class Bolt_O0(nn.Module):
        def forward(self, x):
           return x * (x > 0)


    # def get_max_knob(self):
    #     return (self.max_order,)

    # def decrement_knob(self, knob, i):
    #     knob_new = list(knob)
    #     if knob_new[i] > self.get_min_knob()[i]:
    #         knob_new[i] -= 2
    #         return tuple(knob_new)
    #     else:
    #         return None

    # def increment_knob(self, knob, i):
    #     knob_new = list(knob)
    #     if knob_new[i] < self.get_max_knob()[i]:
    #         knob_new[i] += 2
    #         return tuple(knob_new)
    #     else:
    #         return None

    # def get_min_knob(self):
    #     return (self.min_order,)
    
    def get_possible_knob_positions(self):
        return ((0, 2, 4),)

class ErfPass(BasePass):
    def __init__(self, approx):
        super().__init__(approx=approx)
        self.approx = approx

    def get_patterns(self):
        return [
            lambda x: torch.ops.aten.erf.default(x),
            lambda x: torch.ops.aten.erf_.default(x)
        ]

    def get_replacement(self):
        if self.approx == "poly":
            return self.Poly()
        else:
            raise NotImplementedError()
            

    class Poly(nn.Module):
        def __init__(self):
            super().__init__()
            self.poly_approx = PolyApprox2(coefficents=(
                    -0.001101251074596917, 1.1440618494566452, -0.011568983288181539, -0.48940133506632394, 0.23409163217534779, -0.033644174863832414
                ),
                take_abs=True,
                # scales the input down by 1/3rd to make sure input is 0-1, helps pervent overflow.
                pre_scale=(1/2.5)
            )
        
        def forward(self, x):
            sign_x = x.sign()
            approx, pos_x = self.poly_approx(x, sign_x, return_x=True)
            cond1 = pos_x > 2.33
            return (approx * (~cond1) + cond1) * sign_x

class RsqrtPass(BasePass):
    def __init__(self, approx, T=3):
        super().__init__(approx=approx, T=T)
        self.approx = approx
        self.T = T

    def get_match_filters(self):
        def f(head):
            return is_secret(head.args[0])
        return [f]

    def get_patterns(self):
        return [lambda x: torch.ops.aten.rsqrt.default(x)]

    def get_replacement(self):
        if self.approx == "crypten":
            return self.Crypten(self.T)
        elif self.approx == "crypten-fixed":
            return self.CryptenFixed(self.T)
        elif self.approx == "crypten-fixed-v2":
            return self.CryptenFixedV2(self.T)
        else:
            raise NotImplementedError()

    class Crypten(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

        def forward(self, x):
            y = (-0.5 * x - 0.2).exp() * 2.2 + 0.2 - x / 1024
            for _ in range(self.T):
                y = y * (3 - y * y * x) * 0.5
            return y

    class CryptenFixed(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

        def forward(self, x):
            y = (-0.5 * x - 0.2).exp() * 2.2 + 0.2 - x / 1024

            # Thresholding to avoid diverging
            cond = x < 100
            # y = cond * y + (~cond) * 0.05
            y = torch.where(cond, y, 0.05)
            
            for _ in range(self.T):
                y = y * (3 - y * y * x) * 0.5

            return y

    class CryptenFixedV2(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

        def forward(self, x):
            # Thresholding to avoid diverging
            cond_1 = x < 100
            cond_2 = x < 10_000
            
            pre_factor = torch.where(cond_1, 1.0, torch.where(cond_2, 0.01, 0.0001))
            post_factor = torch.where(cond_1, 1.0,torch.where(cond_2, 0.1, 0.01))
            
            # scale x if too big
            working_x = x * pre_factor
            
            y = (-0.5 * working_x - 0.2).exp() * 2.2 + 0.2 - working_x / 1024

            for _ in range(self.T):
                y = y * (3 - y * y * working_x) * 0.5
            
            # undo the scale if required
            y = y * post_factor

            return y

class TunableGeluPass(TunablePass):
    def __init__(self):
        super().__init__()
        self.max_order = 4
        self.min_order = 0

    def get_patterns(self):
        return [lambda x: torch.ops.aten.gelu.default(x)]

    def get_replacement(self, knobs):
        order = knobs[0]
        if order == 4: 
            return self.Bolt_O4()
        elif order == 2:
            return self.Bolt_O2()
        elif order == 0:
            return self.Bolt_O0()
        else:
            raise NotImplementedError()
    
    class Bolt_O4(nn.Module):
        def forward(self, x):
            g0 = 0.14439048359960427
            g1 = -0.7077117131613893
            g2 = 4.5702822654246535
            g3 = -8.15444702051307
            g4 = 16.382265425072532

            pos_x = x.abs()
            relu_x = (pos_x + x) / 2
            cond = pos_x <= 2.7
            
            gelu_p0 = (g0 * pos_x + g1) * pos_x + g2
            gelu_p1 = (gelu_p0 + g0 * pos_x + g3) * gelu_p0 + g4 + 0.5 * x

            return torch.where(cond, gelu_p1, relu_x)
    
    class Bolt_O2(nn.Module):
        def forward(self, x):
            a0 = -0.03161286950386981
            a1 = 0.2597658446632754
            a2 = 0.11594076368157785

            pos_x = x.abs()
            relu_x = (pos_x + x) / 2
            cond = pos_x <= 2.2
            
            gelu_p0 = pos_x * pos_x * a2 + pos_x * a1 + a0 + 0.5 * x

            return torch.where(cond, gelu_p0, relu_x)
    
    class Bolt_O0(nn.Module):
        def forward(self, x):
           return x * (x > 0)

    def get_possible_knob_positions(self):
        return ((0, 2, 4),)


class TunableExpPass(TunablePass):
    def __init__(self, approx, T=8):
        super().__init__(approx=approx, max_T=T)
        self.approx = approx
        self.max_T = T

    def get_patterns(self):
        return [lambda x: torch.ops.aten.exp.default(x)]

    def get_replacement(self, knobs):
        if self.approx == "crypten-fixed":
            return self.CryptenFixed(knobs)
        else:
            raise NotImplementedError()

    class CryptenFixed(nn.Module):
        def __init__(self, knobs):
            super().__init__()
            self.T = knobs[0]
            self.thres = knobs[1]

        def forward(self, x):
            y = 1 + x / (2 ** self.T)
            # Thresholding to avoid diverging
            # If y is smaller than -1, the output will diverge.
            # Thresholding can be turned of iff comps=(False,) is passed.
            if self.thres == 1:
                y = y * (x > -2 ** (self.T))
            for _ in range(self.T):
                y = y * y
            return y

    def get_match_filters(self):
        def f(head):
            return is_secret(head.args[0])
        return [f]
    
    def get_possible_knob_positions(self):
        return (
            (0, 1, 2, 3, 4, 5, 6, 7, 8), # Iterations
            (0, 1), # clamping or not
        )


class ExpPass(BasePass):
    def __init__(self, approx, T=8):
        super().__init__(approx=approx, T=T)
        self.approx = approx
        self.T = T

    def get_patterns(self):
        return [lambda x: torch.ops.aten.exp.default(x)]

    def get_replacement(self):
        if self.approx == "crypten":
            return self.Crypten(self.T)
        elif self.approx == "crypten-fixed":
            return self.CryptenFixed(self.T)
        else:
            raise NotImplementedError()

    class Crypten(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

        def forward(self, x):
            y = 1 + x / (2 ** self.T)
            for _ in range(self.T):
                y = y * y
            return y

    class CryptenFixed(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

        def forward(self, x):
            y = 1 + x / (2 ** self.T)
            # Thresholding to avoid diverging
            y = y * (x > -600)
            for _ in range(self.T):
                y = y * y
            return y

    def get_match_filters(self):
        def f(head):
            return is_secret(head.args[0])
        return [f]

class SDPAPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        def f(q, k, v, bias):
            y = torch.ops.aten.scaled_dot_product_attention.default(q, k, v, bias)
            return torch.ops.aten.transpose.int(y, 1, 2)
        return [f]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, q, k, v, bias):
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1]) + bias
            attn = scores.softmax(dim=-1)
            y = torch.matmul(attn, v).transpose(1, 2)
            return y.reshape([y.shape[0], y.shape[1], -1])


class SDPANoBiasPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        def f(q, k, v):
            return torch.ops.aten.scaled_dot_product_attention.default(q, k, v)
        return [f]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, q, k, v):
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            attn = scores.softmax(dim=-1)
            return torch.matmul(attn, v)


class MaskedFillPass(BasePass):
    def __init__(self, minval=-100000):
        super().__init__(minval=minval)
        self.minval = minval

    def get_patterns(self):
        return [lambda x, mask, minval: torch.ops.aten.masked_fill.Scalar(x, mask, minval)]

    def get_replacement(self):
        return self.Replacement(self.minval)

    def get_match_filters(self):
        def f(head):
            return head.args[-1] != self.minval
        return [f]

    class Replacement(nn.Module):
        def __init__(self, minval):
            super().__init__()
            self.minval = minval

        def forward(self, x, mask, minval):
            # TODO: Probably decomposing and changing the minval value should be two different things.
            return mask * self.minval + (~mask) * x
            #return mask * self.minval + (1 - mask) * x
            #return x.masked_fill(mask, self.minval)


class SoftmaxPass(BasePass):
    def __init__(self, offset: Optional[int] = None):
        super().__init__(offset=offset)
        self.offset = offset
    
    def get_patterns(self):
        return [lambda x, dim: torch.ops.aten.softmax.int(x, dim)]

    def get_replacement(self):
        return self.Replacement(self.offset)

    class Replacement(nn.Module):
        def __init__(self, offset: Optional[int] = None):
            super().__init__()
            self.offset = offset
        
        def forward(self, x, dim):
            if self.offset is None:
                max_x = torch.amax(x, dim=dim, keepdim=True)
            else:
                max_x = self.offset
            exp = (x - max_x).exp()
            sum = torch.sum(exp, dim=dim, keepdim=True)
            return exp / sum

class LogSoftmaxPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x, dim, half_to_float: torch.ops.aten._log_softmax.default(x, dim, half_to_float)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x, dim, half_to_float):
            max_x = torch.amax(x, dim=dim, keepdim=True)
            logits = x - max_x
            norm_term = (logits).exp().sum(dim=dim, keepdim=True)
            return logits - norm_term.log()


class LogPass(BasePass):
    def __init__(self, approx="crypten", T=2, order=8):
        super().__init__(approx=approx, T=T, order=8)
        self.approx = approx
        self.T = T
        self.order = order
    
    def get_patterns(self):
        return [lambda x: torch.ops.aten.log.default(x)]

    def get_replacement(self):
        if self.approx == "crypten":
            return self.Crypten(self.T, self.order)

    class Crypten(nn.Module):
        def __init__(self, T, order):
            super().__init__()
            self.T = T
            self.poly = PolyApprox(coefficents=[0] + [1 / (i + 1) for i in range(order)])

        def forward(self, x):
            term1 = x / 120
            term2 = (-(2 * x + 1.0)).exp() * 20
            y = term1 - term2 + 3.0

            # Householder iterations
            for _ in range(self.T):
                h = 1 - x * (-y).exp()
                y = y - self.poly(h)
            return y


class DivPass(BasePass):
    def __init__(self):
        super().__init__()
    
    def get_patterns(self):
        return [lambda x, y: torch.ops.aten.div.Tensor(x, y)]

    def get_replacement(self):
        return self.Replacement()

    class Replacement(nn.Module):
        def forward(self, x, y):
            return x * y.reciprocal()

    def get_match_filters(self):
        def f(head):
            #return isinstance(args[1], torch.Tensor)
            return is_secret(head.args[1])
        return [f]


class ReciprocalPass(BasePass):
    def __init__(self, approx, T=10):
        super().__init__(approx=approx, T=T)
        self.approx = approx
        self.T = T

    def get_patterns(self):
        return [lambda x: torch.ops.aten.reciprocal.default(x)]

    def get_replacement(self):
        if self.approx == "crypten":
            return self.Crypten(self.T)

    def get_match_filters(self):
        def f(head):
            return is_secret(head.args[0])
        return [f]

    class Crypten(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

        def forward(self, x):
            sign = x.sign()
            pos_x = x * sign
            y = (-2 * pos_x + 1).exp() * 3 + 0.003
            for _ in range(self.T):
                y = 2 * y - y * y * pos_x
            return y

# Rewriting passes (exact passes, just rewriting).
rewriting_passes = [
    WherePass(),
    ReluPass(),
    GePass(),
    LePass(),
    GtPass(),
    LtPass(),
    AbsPass(),
    SignPass(),
    DivPass(),
    SoftmaxPass(),
    LogSoftmaxPass(),
    SDPAPass(),
    SDPANoBiasPass(),
    LayerNormPass(),
    AddmmPass(),
]
