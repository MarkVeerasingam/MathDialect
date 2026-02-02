#include "MathDialect.h"
#include "MathOps.h"

using namespace mlir;
using namespace mlir::math;

#include "MathDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Math dialect.
//===----------------------------------------------------------------------===//

void MathDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "MathOps.cpp.inc"
      >();
  // registerTypes();
}

Operation *MathDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc)
{
  // This allows the canonicalizer to create a 'math.constant'
  // whenever a fold results in a new value.
  return builder.create<math::ConstantOp>(loc, type, llvm::cast<TypedAttr>(value));
}