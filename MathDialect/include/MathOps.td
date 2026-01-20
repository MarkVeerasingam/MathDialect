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

// Add Op
def Math_AddOp : Math_Op<"add", [Pure]> {
    let summary = "binary addition operation";
    let description = [{
        The "add" operation takes two 32-bit integer operands and returns 
        their sum.
    }];

    // Operands: The two values to be added
    let arguments = (ins I32:$lhs, I32:$rhs);
    
    // Result: The sum of the operands
    let results = (outs I32:$res);

    // %sum = math.add %lhs, %rhs : i32
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

#endif // MATH_OPS