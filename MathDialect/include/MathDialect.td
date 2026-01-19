#ifndef MATH_DIALECT
#define MATH_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Toy Math dialect definition.
//===----------------------------------------------------------------------===//

def Math_Dialect : Dialect {
    let name = "math";
    let summary = "A simple math dialect.";
    let description = [{
        This dialect is a simple math dialect for learning purposes
    }];
    let cppNamespace = "::mlir::math";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base Math operation definition.
//===----------------------------------------------------------------------===//

class Math_Op<string mnemonic, list<Trait> traits = []> :
        Op<Math_Dialect, mnemonic, traits>;

#endif // MATH_DIALECT