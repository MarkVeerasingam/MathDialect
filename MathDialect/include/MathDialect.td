#ifndef MATH_DIALECT
#define MATH_DIALECT

include "mlir/IR/OpBase.td"

def Math_Dialect : Dialect {
    let name = "math";
    let summary = "A math out-of-tree MLIR dialect.";
    let cppNamespace = "::mlir::math";

    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 0;   
}

#endif // MATH_DIALECT