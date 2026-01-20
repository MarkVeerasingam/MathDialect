#include "MathDialect.h"

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
      >();
  // registerTypes();
}