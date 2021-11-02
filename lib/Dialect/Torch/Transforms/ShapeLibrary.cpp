//===-------------------------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//
//
// This file is auto-generated! Do not edit!!!
// Generated with the script `build_tools/update_shape_lib.sh`.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;

StringRef mlir::torch::Torch::getShapeLibrary() {
  constexpr StringLiteral shapeLib(R"mlir(
module  {
  func @"__torch_mlir_shape_fn.aten.tanh"(%arg0: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    return %arg0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.zeros"(%arg0: !torch.list<!torch.int>, %arg1: !torch.optional<!torch.int>, %arg2: !torch.optional<!torch.int>, %arg3: !torch.optional<!torch.Device>, %arg4: !torch.optional<!torch.bool>) -> !torch.list<!torch.int> {
    return %arg0 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.add.Tensor"(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>, %arg2: !torch.float) -> !torch.list<!torch.int> {
    %0 = call @__torch__.broadcast(%arg0, %arg1) : (!torch.list<!torch.int>, !torch.list<!torch.int>) -> !torch.list<!torch.int>
    return %0 : !torch.list<!torch.int>
  }
  func @__torch__.broadcast(%arg0: !torch.list<!torch.int>, %arg1: !torch.list<!torch.int>) -> !torch.list<!torch.int> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %false = torch.constant.bool false
    %str = torch.constant.str "The size of tensor a {} must match the size of tensor b ({}) at non-singleton dimension {}"
    %str_0 = torch.constant.str "AssertionError: "
    %0 = torch.aten.len.t %arg0 : !torch.list<!torch.int> -> !torch.int
    %1 = torch.aten.len.t %arg1 : !torch.list<!torch.int> -> !torch.int
    %2 = torch.prim.max.int %0, %1 : !torch.int, !torch.int -> !torch.int
    %3 = torch.prim.ListConstruct  : () -> !torch.list<!torch.int>
    torch.prim.Loop %2, %true, init()  {
    ^bb0(%arg2: !torch.int):  // no predecessors
      %4 = torch.aten.sub.int %2, %int1 : !torch.int, !torch.int -> !torch.int
      %5 = torch.aten.sub.int %4, %arg2 : !torch.int, !torch.int -> !torch.int
      %6 = torch.aten.sub.int %0, %int1 : !torch.int, !torch.int -> !torch.int
      %7 = torch.aten.sub.int %6, %5 : !torch.int, !torch.int -> !torch.int
      %8 = torch.aten.sub.int %1, %int1 : !torch.int, !torch.int -> !torch.int
      %9 = torch.aten.sub.int %8, %5 : !torch.int, !torch.int -> !torch.int
      %10 = torch.aten.ge.int %7, %int0 : !torch.int, !torch.int -> !torch.bool
      %11 = torch.prim.If %10 -> (!torch.int) {
        %20 = torch.aten.__getitem__.t %arg0, %7 : !torch.list<!torch.int>, !torch.int -> !torch.int
        torch.prim.If.yield %20 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %12 = torch.aten.ge.int %9, %int0 : !torch.int, !torch.int -> !torch.bool
      %13 = torch.prim.If %12 -> (!torch.int) {
        %20 = torch.aten.__getitem__.t %arg1, %9 : !torch.list<!torch.int>, !torch.int -> !torch.int
        torch.prim.If.yield %20 : !torch.int
      } else {
        torch.prim.If.yield %int1 : !torch.int
      }
      %14 = torch.aten.ne.int %11, %13 : !torch.int, !torch.int -> !torch.bool
      %15 = torch.prim.If %14 -> (!torch.bool) {
        %20 = torch.aten.ne.int %11, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      %16 = torch.prim.If %15 -> (!torch.bool) {
        %20 = torch.aten.ne.int %13, %int1 : !torch.int, !torch.int -> !torch.bool
        torch.prim.If.yield %20 : !torch.bool
      } else {
        torch.prim.If.yield %false : !torch.bool
      }
      torch.prim.If %16 -> () {
        %20 = torch.aten.format(%str, %11, %13, %arg2) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
        %21 = torch.aten.add.str %str_0, %20 : !torch.str, !torch.str -> !torch.str
        torch.prim.RaiseException %21 : !torch.str
        torch.prim.If.yield
      } else {
        torch.prim.If.yield
      }
      %17 = torch.aten.eq.int %11, %int1 : !torch.int, !torch.int -> !torch.bool
      %18 = torch.prim.If %17 -> (!torch.int) {
        torch.prim.If.yield %13 : !torch.int
      } else {
        torch.prim.If.yield %11 : !torch.int
      }
      %19 = torch.aten.append.t %3, %18 : !torch.list<!torch.int>, !torch.int -> !torch.list<!torch.int>
      torch.prim.Loop.condition %true, iter()
    } : (!torch.int, !torch.bool) -> ()
    return %3 : !torch.list<!torch.int>
  }
  func @"__torch_mlir_shape_fn.aten.topk"(%arg0: !torch.list<!torch.int>, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.bool, %arg4: !torch.bool) -> !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>> {
    %str = torch.constant.str "k ({}) is too big for dimension {} of size {}"
    %str_0 = torch.constant.str "AssertionError: "
    %0 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
    %1 = torch.aten.le.int %arg1, %0 : !torch.int, !torch.int -> !torch.bool
    torch.prim.If %1 -> () {
      torch.prim.If.yield
    } else {
      %4 = torch.aten.__getitem__.t %arg0, %arg2 : !torch.list<!torch.int>, !torch.int -> !torch.int
      %5 = torch.aten.format(%str, %arg1, %arg2, %4) : !torch.str, !torch.int, !torch.int, !torch.int -> !torch.str
      %6 = torch.aten.add.str %str_0, %5 : !torch.str, !torch.str -> !torch.str
      torch.prim.RaiseException %6 : !torch.str
      torch.prim.If.yield
    }
    %2 = torch.aten._set_item.t %arg0, %arg2, %arg1 : !torch.list<!torch.int>, !torch.int, !torch.int -> !torch.list<!torch.int>
    %3 = torch.prim.TupleConstruct %arg0, %arg0 : !torch.list<!torch.int>, !torch.list<!torch.int> -> !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>>
    return %3 : !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>>
  }
}
)mlir");
  return shapeLib;
}
