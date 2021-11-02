//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

// Populates the shape calculation region with a call to the shape function
// from the shape library.
static void populateShapeCalculationRegion(ShapeCalculateOp op,
                                           ValueRange originalOperands,
                                           mlir::FuncOp shapeFunction) {
  // Create a call to the shape function in the `shapeCalculation` region.
  // We will import the callee from the shape library later.
  OpBuilder b(op.getContext());
  Location loc = op->getLoc();
  b.createBlock(&op.shapeCalculation());
  // Massage the op operands to match the shape function signature.
  // The shape function generally takes the same operands as the op, with a few
  // systematic modifications, such as replacing tensors with their shapes.
  SmallVector<Value> shapeFunctionArgs;
  for (auto operandAndDesiredType :
       llvm::zip(originalOperands, shapeFunction.getArgumentTypes())) {
    Value operand;
    Type desiredType;
    std::tie(operand, desiredType) = operandAndDesiredType;

    // The shape library functions have tensor operands replaced with
    // `!torch.list<!torch.int>` types for the shape. Get the sizes.
    if (operand.getType().isa<Torch::BaseTensorType>()) {
      auto adjusted = b.create<AtenSizeOp>(loc, desiredType, operand);
      shapeFunctionArgs.push_back(adjusted);
      continue;
    }
    // The shape library functions use `float` where the operator
    // signature uses `Scalar` (see comments in torch_ods_gen.py for
    // explanation), so adjust the type here.
    if (desiredType.isa<Torch::FloatType>() &&
        operand.getType().isa<Torch::IntType>()) {
      auto adjusted = b.create<AtenFloatScalarOp>(loc, desiredType, operand);
      shapeFunctionArgs.push_back(adjusted);
      continue;
    }
    // Pass the operand as-is.
    shapeFunctionArgs.push_back(operand);
  }

  // Create the call to the shape function!
  auto call = b.create<mlir::CallOp>(loc, shapeFunction, shapeFunctionArgs);

  // Python models multiple results with a tuple, so we need to unpack it
  // if the op has multiple results.
  SmallVector<Value> unpackedResults;
  assert(call.getNumResults() == 1 &&
         "Multiple results are packed in a tuple in Python!");
  Value result = call.getResult(0);
  if (auto tupleType = result.getType().dyn_cast<Torch::TupleType>()) {
    auto unpack =
        b.create<PrimTupleUnpackOp>(loc, tupleType.getContainedTypes(), result);
    llvm::append_range(unpackedResults, unpack.getResults());
  } else {
    unpackedResults.push_back(result);
  }

  // Terminate the region.
  b.create<ShapeCalculateYieldShapesOp>(loc, unpackedResults);
}

namespace {
class ReifyShapeCalculationsPass
    : public ReifyShapeCalculationsBase<ReifyShapeCalculationsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    // TODO: Find a way to not have to parse this every time.
    // The shape library is O(#ops we know about), and this pass should be
    // O(#ops in the program) ideally.
    auto shapeLibrary = parseSourceString(getShapeLibrary(), context);

    // Walk all the operations, and if we have a shape function, wrap the op
    // in a `torch.shape.calculate` op.
    SmallVector<std::string> neededShapeFunctions;
    module.walk([&](Operation *op) {
      Location loc = op->getLoc();
      auto name = op->getName().stripDialect();
      auto shapeFunctionName = ("__torch_mlir_shape_fn." + Twine(name)).str();
      auto shapeFunction =
          shapeLibrary->lookupSymbol<FuncOp>(shapeFunctionName);
      if (!shapeFunction)
        return;
      neededShapeFunctions.push_back(shapeFunctionName);
      auto shapeCalculate =
          OpBuilder(op).create<ShapeCalculateOp>(loc, op->getResultTypes());
      op->replaceAllUsesWith(shapeCalculate);
      {
        // Move the op into the body of the `torch.shape.calculate` op and yield
        // its results.
        OpBuilder b(context);
        Block *block = b.createBlock(&shapeCalculate.body());
        op->moveBefore(block, block->end());
        b.setInsertionPointAfter(op);
        b.create<ShapeCalculateYieldOp>(loc, op->getResults());
      }
      populateShapeCalculationRegion(shapeCalculate, op->getOperands(),
                                     shapeFunction);
    });

    // Import just the functions we need. This includes transitive callees,
    // so we use a worklist algorithm.
    llvm::StringSet<> importedFunctions;
    SmallVector<std::string> worklist;
    llvm::append_range(worklist, neededShapeFunctions);
    while (!worklist.empty()) {
      auto symName = worklist.pop_back_val();
      if (importedFunctions.count(symName))
        continue;
      auto func = shapeLibrary->lookupSymbol<mlir::FuncOp>(symName);
      assert(func && "broken shape library");
      // Move the shape function from the library to the module this pass
      // is running on. (this mutates the library, but we re-parse it each time
      // so this is safe to do).
      func->moveBefore(&module.getBody()->front());
      // Set the visibility to private so that the shape functions go away
      // nicely after we are done with them.
      func.setVisibility(SymbolTable::Visibility::Private);
      // Continue the DFS.
      importedFunctions.insert(symName);
      func.walk([&](CallOp op) { worklist.push_back(op.callee().str()); });
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::torch::Torch::createReifyShapeCalculationsPass() {
  return std::make_unique<ReifyShapeCalculationsPass>();
}
