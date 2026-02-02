#include "MathOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::math;

// // === ConstantOp Implementation ===

// OpFoldResult ConstantOp::fold(FoldAdaptor adaptor)
// {
//   // Simply return the attribute value itself
//   return getValue();
// }

// LogicalResult ConstantOp::verify()
// {
//   // Basic verification: ensure the value attribute matches the result type
//   auto type = getType();
//   auto value = getValue();

//   if (!llvm::isa<IntegerAttr, FloatAttr, ElementsAttr>(value))
//     return emitOpError("requires an integer or floating point attribute");

//   return success();
// }

// ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result)
// {
//   Attribute valueAttr;
//   // This parses "5 : i32" as a single TypedAttr
//   if (parser.parseAttribute(valueAttr))
//     return failure();

//   result.addAttribute("value", valueAttr);

//   // Extract the type from the attribute to set the result type
//   if (auto typedAttr = llvm::dyn_cast<TypedAttr>(valueAttr))
//     result.addTypes(typedAttr.getType());

//   return success();
// }

// void ConstantOp::print(OpAsmPrinter &p)
// {
//   p << " ";
//   p.printAttribute(getValue()); // This prints "5 : i32"
// }

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

#define GET_OP_CLASSES
#include "MathOps.cpp.inc"