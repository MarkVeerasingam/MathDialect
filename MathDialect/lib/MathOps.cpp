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

//===----------------------------------------------------------------------===//
// MulOp Folding
//===----------------------------------------------------------------------===//

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  // 1. Identity: x * 1 = x
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  // 2. Identity: x * 0 = 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs(); // Returns the zero attribute

  // 3. Constant Folding
  auto lhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return IntegerAttr::get(lhs.getType(), lhs.getInt() * rhs.getInt());

  return nullptr;
}

//===----------------------------------------------------------------------===//
// DivOp Folding
//===----------------------------------------------------------------------===//

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  // 1. Identity: x / 1 = x
  if (matchPattern(getRhs(), m_One()))
    return getLhs();

  // 2. Constant Folding
  auto lhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getLhs());
  auto rhs = llvm::dyn_cast_if_present<IntegerAttr>(adaptor.getRhs());
  
  if (lhs && rhs) {
    auto divisor = rhs.getInt();
    // 3. Safety Check: Don't fold if divisor is 0!
    // If we don't check this, the compiler itself will crash (SIGFPE)
    if (divisor != 0)
      return IntegerAttr::get(lhs.getType(), lhs.getInt() / divisor);
  }

  return nullptr;
}

#define GET_OP_CLASSES
#include "MathOps.cpp.inc"