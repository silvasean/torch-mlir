# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch

from torch_mlir_e2e_test.torchscript.framework import TestUtils
from torch_mlir_e2e_test.torchscript.registry import register_test_case
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

# TODO: Support scalar !torch.int/!torch.float variants. Add support to
# ReduceOpVariants to implement them in terms of the tensor-only variants +
# torch.prim.NumToTensor.

# TODO: This is pretty verbose. Can we have a helper to reduce
# the boilerplate?

# ==============================================================================


class ElementwiseUnaryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        return torch.tanh(a)


@register_test_case(module_factory=lambda: ElementwiseUnaryModule())
def ElementwiseUnaryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4))


# ==============================================================================


class ElementwiseBinaryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b


@register_test_case(module_factory=lambda: ElementwiseBinaryModule())
def ElementwiseBinaryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4), tu.rand(4))


# ==============================================================================


class ElementwiseTernaryModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1], torch.float32, True),
    ])
    def forward(self, a, b, c):
        return torch.lerp(a, b, c)


@register_test_case(module_factory=lambda: ElementwiseTernaryModule())
def ElementwiseTernaryModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 4, 5), tu.rand(4, 5), tu.rand(5))


# ==============================================================================


# Addition is an interesting special case of a binary op, because under the hood
# it carries a third scalar "alpha" parameter, which needs special handling.
class ElementwiseAddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return a + b


@register_test_case(module_factory=lambda: ElementwiseAddModule())
def ElementwiseAddModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand())


# ==============================================================================


class ElementwiseUnsqueezeBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b.unsqueeze(0)


@register_test_case(
    module_factory=lambda: ElementwiseUnsqueezeBroadcastModule())
def ElementwiseUnsqueezeBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4), tu.rand())


# ==============================================================================


class ElementwiseUnsqueezeNegDimsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, a):
        # As mentioned in `unsqueeze` docstring,
        # valid dim values are [-input.dim()-1, input.dim()+1).
        # This tests the lower bound
        return torch.unsqueeze(a, -3)


@register_test_case(
    module_factory=lambda: ElementwiseUnsqueezeNegDimsModule())
def ElementwiseUnsqueezeNegDimsModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 3))


# ==============================================================================


class ElementwiseFlattenBroadcastModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1], torch.float32, True),
        ([], torch.float32, True),
    ])
    def forward(self, a, b):
        return a * b.flatten(-1, -1)


@register_test_case(module_factory=lambda: ElementwiseFlattenBroadcastModule())
def ElementwiseFlattenBroadcastModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(6), tu.rand())

# ==============================================================================


class ElementwiseReluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.relu(x)


@register_test_case(module_factory=lambda: ElementwiseReluModule())
def ElementwiseReluModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(4, 2) - 0.5)

# ==============================================================================


class ElementwiseSigmoidModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return torch.sigmoid(x)


@register_test_case(module_factory=lambda: ElementwiseSigmoidModule())
def ElementwiseSigmoidModule_basic(module, tu: TestUtils):
    module.forward(tu.rand(3, 5))

# ==============================================================================
