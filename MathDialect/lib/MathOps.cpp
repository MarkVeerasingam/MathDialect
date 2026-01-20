#include "MathOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::math;

#define GET_OP_CLASSES
#include "MathOps.cpp.inc"