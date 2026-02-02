#include "MathOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::math;

// === ConstantOp Implementation ===

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor)
{
  // Simply return the attribute value itself
  return getValue();
}

LogicalResult ConstantOp::verify()
{
  // Basic verification: ensure the value attribute matches the result type
  auto type = getType();
  auto value = getValue();

  if (!llvm::isa<IntegerAttr, FloatAttr>(value))
    return emitOpError("requires an integer or floating point attribute");

  return success();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result)
{
  Attribute valueAttr;
  Type type;
  // Parses: 10 : i32  OR  1.0 : f32
  if (parser.parseAttribute(valueAttr) || parser.parseColonType(type))
    return failure();

  result.addAttribute("value", valueAttr);
  result.addTypes(type);
  return success();
}

void ConstantOp::print(OpAsmPrinter &p)
{
  p << " ";
  p.printAttributeWithoutType(getValue());
  p << " : ";
  p.printType(getType());
}

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

#define GET_OP_CLASSES
#include "MathOps.cpp.inc"