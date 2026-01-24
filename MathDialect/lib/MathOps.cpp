#include "MathOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::math;

/**
 * dev note: this is modeled from arith: https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Arith/IR/ArithOps.cpp
 */

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
    return getValueAttr();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // x + 0 = x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();

  // (a - b) + b = a
  if (auto subOp = getLhs().getDefiningOp<SubOp>()) {
    if (subOp.getRhs() == getRhs())
      return subOp.getLhs();
  }

  // Constant Folding
  auto lhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (lhs && rhs) {
    return IntegerAttr::get(lhs.getType(), lhs.getInt() + rhs.getInt());
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
    // x - 0 = x
    if (matchPattern(getRhs(), m_Zero()))
        return getLhs();

    // x - x = 0
    if (getLhs() == getRhs())
        return IntegerAttr::get(getLhs().getType(), 0);

    // Constant Folding
    auto lhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
    auto rhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
    if (lhs && rhs) {
        return IntegerAttr::get(lhs.getType(), lhs.getInt() - rhs.getInt());
    }

    return nullptr;
}

#define GET_OP_CLASSES
#include "MathOps.cpp.inc"