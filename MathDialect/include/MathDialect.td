include "mlir/IR/OpBase.td"

def Math_Dialect : Dialect {
    let name = "math";
    let summary = "A math MLIR dialect.";
    let cppNamespace = "::mlir::math";

    let hasConstantMaterializer = 1;

    let useDefaultTypePrinterParser = 0;
    let useDefaultAttributePrinterParser = 0;   
}