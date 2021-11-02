// RUN: torch-mlir-opt -torch-reify-shape-calculations -split-input-file %s | FileCheck %s

// CHECK: module {
// CHECK: func private @__torch_mlir_shape_fn.aten.tanh(

// CHECK-LABEL:   func @basic(
// CHECK-SAME:                %[[ARG:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate  {
// CHECK:             %[[TANH:.*]] = torch.aten.tanh %[[ARG]] : !torch.vtensor -> !torch.vtensor
// CHECK:             torch.shape.calculate.yield %[[TANH]] : !torch.vtensor
// CHECK:           } shapes  {
// CHECK:             %[[SHAPE:.*]] = torch.aten.size %[[ARG]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[RESULT_SHAPE:.*]] = call @__torch_mlir_shape_fn.aten.tanh(%[[SHAPE]]) : (!torch.list<!torch.int>) -> !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[RESULT_SHAPE]] : !torch.list<!torch.int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func @basic(%arg0: !torch.vtensor) -> !torch.vtensor {
  %0 = torch.aten.tanh %arg0 : !torch.vtensor -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// torch.aten.add.Tensor also has a shape function that calls a "broadcast"
// helper, so this test also checks our logic for transitively pulling in
// callees of the shape functions.

// CHECK: module {
// CHECK: func private @__torch_mlir_shape_fn.aten.add.Tensor(

// CHECK-LABEL:   func @scalar_args(
// CHECK-SAME:                      %[[ARG0:.*]]: !torch.vtensor,
// CHECK-SAME:                      %[[ARG1:.*]]: !torch.vtensor) -> !torch.vtensor {
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULT:.*]] = torch.shape.calculate  {
// CHECK:             %[[ADD:.*]] = torch.aten.add.Tensor %[[ARG0]], %[[ARG1]], %[[INT1]] : !torch.vtensor, !torch.vtensor, !torch.int -> !torch.vtensor
// CHECK:             torch.shape.calculate.yield %[[ADD]] : !torch.vtensor
// CHECK:           } shapes  {
// CHECK:             %[[ARG0_SHAPE:.*]] = torch.aten.size %[[ARG0]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[ARG1_SHAPE:.*]] = torch.aten.size %[[ARG1]] : !torch.vtensor -> !torch.list<!torch.int>
// CHECK:             %[[SCALAR_CONVERTED:.*]] = torch.aten.Float.Scalar %[[INT1]] : !torch.int -> !torch.float
// CHECK:             %[[RESULT_SHAPE:.*]] = call @__torch_mlir_shape_fn.aten.add.Tensor(%[[ARG0_SHAPE]], %[[ARG1_SHAPE]], %[[SCALAR_CONVERTED]]) : (!torch.list<!torch.int>, !torch.list<!torch.int>, !torch.float) -> !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[RESULT_SHAPE]] : !torch.list<!torch.int>
// CHECK:           } : !torch.vtensor
// CHECK:           return %[[RESULT:.*]] : !torch.vtensor
func @scalar_args(%arg0: !torch.vtensor, %arg1: !torch.vtensor) -> !torch.vtensor {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor, !torch.vtensor, !torch.int -> !torch.vtensor
  return %0 : !torch.vtensor
}

// -----

// CHECK: module {
// CHECK: func private @__torch_mlir_shape_fn.aten.topk(

// CHECK-LABEL:   func @multiple_results(
// CHECK-SAME:                           %[[ARG:.*]]: !torch.tensor) -> (!torch.tensor, !torch.tensor) {
// CHECK:           %[[TRUE:.*]] = torch.constant.bool true
// CHECK:           %[[INT3:.*]] = torch.constant.int 3
// CHECK:           %[[INT1:.*]] = torch.constant.int 1
// CHECK:           %[[RESULTS:.*]]:2 = torch.shape.calculate  {
// CHECK:             %[[TOP_VALUES:.*]], %[[TOPK_INDICES:.*]] = torch.aten.topk %[[ARG]], %[[INT3]], %[[INT1]], %[[TRUE]], %[[TRUE]] : !torch.tensor, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.tensor, !torch.tensor
// CHECK:             torch.shape.calculate.yield %[[TOP_VALUES]], %[[TOPK_INDICES]] : !torch.tensor, !torch.tensor
// CHECK:           } shapes  {
// CHECK:             %[[ARG_SHAPE:.*]] = torch.aten.size %[[ARG]] : !torch.tensor -> !torch.list<!torch.int>
// CHECK:             %[[TOPK_SHAPE_TUPLE:.*]] = call @__torch_mlir_shape_fn.aten.topk(%[[ARG_SHAPE]], %[[INT3]], %[[INT1]], %[[TRUE]], %[[TRUE]]) : (!torch.list<!torch.int>, !torch.int, !torch.int, !torch.bool, !torch.bool) -> !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>>
// CHECK:             %[[TOPK_SHAPE:.*]]:2 = torch.prim.TupleUnpack %[[TOPK_SHAPE_TUPLE]] : !torch.tuple<!torch.list<!torch.int>, !torch.list<!torch.int>> -> !torch.list<!torch.int>, !torch.list<!torch.int>
// CHECK:             torch.shape.calculate.yield.shapes %[[TOPK_SHAPE]]#0, %[[TOPK_SHAPE]]#1 : !torch.list<!torch.int>, !torch.list<!torch.int>
// CHECK:           } : !torch.tensor, !torch.tensor
// CHECK:           return %[[RESULTS:.*]]#0, %[[RESULTS]]#1 : !torch.tensor, !torch.tensor

func @multiple_results(%arg0: !torch.tensor) -> (!torch.tensor, !torch.tensor) {
  %true = torch.constant.bool true
  %int3 = torch.constant.int 3
  %int1 = torch.constant.int 1
  %values, %indices = torch.aten.topk %arg0, %int3, %int1, %true, %true : !torch.tensor, !torch.int, !torch.int, !torch.bool, !torch.bool -> !torch.tensor, !torch.tensor
  return %values, %indices : !torch.tensor, !torch.tensor
}
