#include "MathOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::math;

// === AddOp Implementation ===
OpFoldResult AddOp::fold(FoldAdaptor adaptor)
{
  // 1. Algebraic simplification: x + 0 = x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  // 2. Constant folding
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (lhs && rhs)
  {
    // Integer Case
    if (auto lInt = llvm::dyn_cast<IntegerAttr>(lhs))
    {
      if (auto rInt = llvm::dyn_cast<IntegerAttr>(rhs))
      {
        return IntegerAttr::get(lInt.getType(), lInt.getValue() + rInt.getValue());
      }
    }

    // Floating Point Case
    if (auto lFloat = llvm::dyn_cast<FloatAttr>(lhs))
    {
      if (auto rFloat = llvm::dyn_cast<FloatAttr>(rhs))
      {
        APFloat result = lFloat.getValue();
        result.add(rFloat.getValue(), APFloat::rmNearestTiesToEven);
        return FloatAttr::get(lFloat.getType(), result);
      }
    }
  }

  return nullptr;
}

// Pattern: x + 0 => x
struct SimplifyAddZero : public OpRewritePattern<AddOp>
{
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override
  {
    if (matchPattern(op.getRhs(), m_Zero()))
    {
      rewriter.replaceOp(op, op.getLhs());
      return success();
    }
    return failure();
  }
};

// Pattern: math.add(%tensor, math.splat(%scalar)) -> math.add_ts(%tensor, %scalar)
struct FuseAddSplat : public OpRewritePattern<AddOp>
{
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const override
  {
    // We expect the RHS to be the result of a SplatOp
    auto splatOp = op.getRhs().getDefiningOp<SplatOp>();
    if (!splatOp)
      return failure();

    // Replace the generic Add with your specialized AddTensorScalar
    rewriter.replaceOpWithNewOp<AddTensorScalarOp>(
        op,
        op.getType(),    // Result type
        op.getLhs(),     // The Tensor
        splatOp.getSrc() // The raw Scalar (un-splatted!)
    );
    return success();
  }
};

// === SubOp Implementation ===
OpFoldResult SubOp::fold(FoldAdaptor adaptor)
{
  // x - 0 = x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs)
    return nullptr;

  if (auto lInt = llvm::dyn_cast<IntegerAttr>(lhs))
  {
    auto rInt = llvm::dyn_cast<IntegerAttr>(rhs);
    return IntegerAttr::get(lInt.getType(), lInt.getValue() - rInt.getValue());
  }
  if (auto lFloat = llvm::dyn_cast<FloatAttr>(lhs))
  {
    auto rFloat = llvm::dyn_cast<FloatAttr>(rhs);
    APFloat result = lFloat.getValue();
    result.subtract(rFloat.getValue(), APFloat::rmNearestTiesToEven);
    return FloatAttr::get(lFloat.getType(), result);
  }
  return nullptr;
}

// Pattern: x - x  => 0
struct SimplifySameOperandSub : public OpRewritePattern<SubOp>
{
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op, PatternRewriter &rewriter) const override
  {
    if (op.getLhs() != op.getRhs())
      return failure();

    // Replace with a constant zero of the same type
    auto zeroAttr = rewriter.getZeroAttr(op.getType());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, zeroAttr);
    return success();
  }
};

// === MulOp Implementation ===
OpFoldResult MulOp::fold(FoldAdaptor adaptor)
{
  // x * 1 = x
  if (matchPattern(getRhs(), m_One()))
    return getLhs();
  // x * 0 = 0 (Check for side effects? No, we are Pure)
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();

  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs)
    return nullptr;

  if (auto lInt = llvm::dyn_cast<IntegerAttr>(lhs))
  {
    auto rInt = llvm::dyn_cast<IntegerAttr>(rhs);
    return IntegerAttr::get(lInt.getType(), lInt.getValue() * rInt.getValue());
  }
  if (auto lFloat = llvm::dyn_cast<FloatAttr>(lhs))
  {
    auto rFloat = llvm::dyn_cast<FloatAttr>(rhs);
    APFloat result = lFloat.getValue();
    result.multiply(rFloat.getValue(), APFloat::rmNearestTiesToEven);
    return FloatAttr::get(lFloat.getType(), result);
  }
  return nullptr;
}

// === DivOp Implementation ===
OpFoldResult DivOp::fold(FoldAdaptor adaptor)
{
  // x / 1 = x
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();
  if (!lhs || !rhs)
    return nullptr;

  // Integer division
  if (auto lInt = llvm::dyn_cast<IntegerAttr>(lhs))
  {
    auto rInt = llvm::dyn_cast<IntegerAttr>(rhs);
    APInt rVal = rInt.getValue();
    if (rVal.isZero())
      return nullptr; // Don't fold div by zero!
    return IntegerAttr::get(lInt.getType(), lInt.getValue().sdiv(rVal));
  }

  // Float division
  if (auto lFloat = llvm::dyn_cast<FloatAttr>(lhs))
  {
    auto rFloat = llvm::dyn_cast<FloatAttr>(rhs);
    APFloat rVal = rFloat.getValue();
    // In floats, we check if divisor is zero to avoid folding to Infinity/NaN
    // unless that's your specific dialect's desired behavior.
    if (rVal.isZero())
      return nullptr;
    APFloat result = lFloat.getValue();
    result.divide(rVal, APFloat::rmNearestTiesToEven);
    return FloatAttr::get(lFloat.getType(), result);
  }
  return nullptr;
}

OpFoldResult AddTensorScalarOp::fold(FoldAdaptor adaptor)
{
  // For now, we return nullptr so the linker is happy.
  // Later, we can implement logic to fold if both the
  // tensor and scalar are constants.
  return nullptr;
}

// === SplatOp Implementation ===
OpFoldResult SplatOp::fold(FoldAdaptor adaptor)
{
  // Use the adaptor to get the attribute of the 'src' operand
  auto srcAttr = adaptor.getSrc();
  if (!srcAttr)
    return nullptr;

  // Get the result type (the tensor type)
  auto tensorType = llvm::cast<RankedTensorType>(getResult().getType());

  // Create a SplatElementsAttr: this is the "canonical" way to represent
  // a tensor filled with a single constant value.
  return SplatElementsAttr::get(tensorType, srcAttr);
}

// Register pattern rewrite
void SubOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<SimplifySameOperandSub>(context);
}

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)
{
  results.add<SimplifyAddZero>(context);
}

#define GET_OP_CLASSES
#include "MathOps.cpp.inc"