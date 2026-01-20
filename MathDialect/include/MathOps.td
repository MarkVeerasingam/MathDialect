#ifndef MATH_OPS
#define MATH_OPS

include "MathDialect.td" 
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all Math ops
class Math_Op<string mnemonic, list<Trait> traits = []> :
    Op<Math_Dialect, mnemonic, traits>;

// Constant Op
def Math_ConstantOp : Math_Op<"constant", [ConstantLike, Pure]> {
    let arguments = (ins I32Attr:$value);
    let results = (outs I32:$res);
    let assemblyFormat = "$value attr-dict `:` type($res)";
}

#endif // MATH_OPS