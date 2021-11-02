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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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

namespace {
// TODO: Only unroll inside the shape calculation region.
// Maybe do this by only applying patterns and folding greedily on the ops
// inside the region + the shape.calculate op itself?
class FullyUnrollPrimLoopOp : public OpRewritePattern<PrimLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimLoopOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *context = op->getContext();
    if (!op.isForLike())
      return failure();
    int64_t maxTripCount;
    if (!matchPattern(op.maxTripCount(), m_TorchConstantInt(&maxTripCount)))
      return failure();
    SmallVector<Value> indices;
    for (int64_t i = 0; i < maxTripCount; i++) {
      // TODO: Add convenience builder.
      indices.push_back(rewriter.create<ConstantIntOp>(
          loc, rewriter.getIntegerAttr(IntegerType::get(context, 64), i)));
    }
    Block *beforeBlock = op->getBlock();
    Block *afterBlock = rewriter.splitBlock(op->getBlock(), op->getIterator());

    SmallVector<Block *> blocksToMerge;
    BlockAndValueMapping bvm;
    // TODO: Helper for region().front()
    auto condition =
        cast<PrimLoopConditionOp>(op.region().front().getTerminator());
    for (int64_t i = 0; i < maxTripCount; i++) {
      SmallVector<Value> iterArgs;
      if (i == 0) {
        llvm::append_range(iterArgs, op.iterArgsInit());
      } else {
        llvm::append_range(
            iterArgs, llvm::map_range(condition.iterArgs(),
                                      [&](Value v) { return bvm.lookup(v); }));
      }
      bvm.clear();
      bvm.map(op.region().front().getArgument(0), indices[i]);
      bvm.map(op.region().front().getArguments().slice(1), iterArgs);

      op.region().cloneInto(afterBlock->getParent(), afterBlock->getIterator(),
                            bvm);
      Block *clonedBlock = bvm.lookup(&op.region().front());
      rewriter.eraseOp(clonedBlock->getTerminator());
      blocksToMerge.push_back(clonedBlock);
    }

    blocksToMerge.push_back(afterBlock);
    for (Block *block : blocksToMerge)
      rewriter.mergeBlocks(block, beforeBlock);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
class AbstractlyInterpretListOpsWithinABlock
    : public OpRewritePattern<PrimListConstructOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimListConstructOp op,
                                PatternRewriter &rewriter) const override {
    Block *block = op->getBlock();
    auto users = llvm::to_vector<6>(op->getUsers());
    for (Operation *user : users) {
      if (user->getBlock() != block)
        return failure();
    }
    llvm::sort(users, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });

    SmallVector<SmallVector<Value>> listLiterals;
    SmallVector<Value> runningList;
    llvm::append_range(runningList, op->getOperands());
    for (Operation *user : users) {
      if (auto append = dyn_cast<AtenAppendTOp>(user)) {
        if (!append.use_empty())
          return failure();
        runningList.push_back(append.el());
        listLiterals.push_back(runningList);
        continue;
      }
      break;
    }
    if (listLiterals.empty())
      return failure();

    Value latestLiteral = nullptr;
    for (const auto &t : llvm::zip(
             makeArrayRef(users).slice(0, listLiterals.size()), listLiterals)) {
      Operation *user = std::get<0>(t);
      auto listLiteral = std::get<1>(t);
      rewriter.setInsertionPoint(user);
      latestLiteral = rewriter.replaceOpWithNewOp<PrimListConstructOp>(
          user, user->getResultTypes()[0], listLiteral);
    }
    rewriter.replaceOp(op, latestLiteral);

    return success();
  }
};
} // namespace

namespace {
class SimplifyShapeCalculationsPass
    : public SimplifyShapeCalculationsBase<SimplifyShapeCalculationsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.insert<FullyUnrollPrimLoopOp>(context);
    patterns.insert<AbstractlyInterpretListOpsWithinABlock>(context);

    PrimIfOp::getCanonicalizationPatterns(patterns, context);
    Aten__Getitem__TOp::getCanonicalizationPatterns(patterns, context);
    AtenSizeOp::getCanonicalizationPatterns(patterns, context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      getOperation().dump();
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::Torch::createSimplifyShapeCalculationsPass() {
  return std::make_unique<SimplifyShapeCalculationsPass>();
}
