include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MathDialect.td"

// Base class
class Math_Op<string mnemonic, list<Trait> traits = []> :
    Op<Math_Dialect, mnemonic, traits>;

// // Constant Op
// def Math_ConstantOp : Math_Op<"constant", [Pure, ConstantLike]> {
//     let summary = "Constant value (scalar or tensor)";
//     let arguments = (ins AnyAttr:$value);
    
//     // Allow the result to be a scalar OR a ranked tensor
//     let results = (outs AnyTypeOf<[AnyFloat, AnyInteger, AnyRankedTensor]>:$res);
    
//     let hasCustomAssemblyFormat = 1;
//     let hasFolder = 1;
//     let hasVerifier = 1;
// }

// Add Op
def Math_AddOp : Math_Op<"add", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let arguments = (ins AnyTypeOf<[AnyFloat, AnyInteger]>:$lhs, 
                         AnyTypeOf<[AnyFloat, AnyInteger]>:$rhs);

    let results = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$res);

    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
    let hasFolder = 1; 
}

// Subtraction Op
def Math_SubOp : Math_Op<"sub", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let arguments = (ins AnyTypeOf<[AnyFloat, AnyInteger]>:$lhs, 
                         AnyTypeOf<[AnyFloat, AnyInteger]>:$rhs);
    let results = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
    let hasFolder = 1;
}

// Multiplication Op
def Math_MulOp : Math_Op<"mul", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let arguments = (ins AnyTypeOf<[AnyFloat, AnyInteger]>:$lhs, 
                         AnyTypeOf<[AnyFloat, AnyInteger]>:$rhs);
    let results = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
    let hasFolder = 1;
}

// Division Op
def Math_DivOp : Math_Op<"div", [Pure, AllTypesMatch<["lhs", "rhs", "res"]>]> {
    let arguments = (ins AnyTypeOf<[AnyFloat, AnyInteger]>:$lhs, 
                         AnyTypeOf<[AnyFloat, AnyInteger]>:$rhs);
    let results = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$res);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
    let hasFolder = 1;
}

// Splat Op
def Math_SplatOp : Math_Op<"splat", [
    Pure,
    // Move the matching logic here as a Trait
    TypesMatchWith<"scalar type matches tensor element type",
                   "result", "src",
                   "llvm::cast<TensorType>($_self).getElementType()">
  ]> {
    let summary = "Splat a scalar value into a tensor";
    let description = [{
        The `math.splat` operation creates a tensor where every element is 
        equal to the input scalar.
    }];

    let arguments = (ins AnyTypeOf<[AnyFloat, AnyInteger]>:$src);
    let results = (outs RankedTensorOf<[AnyFloat, AnyInteger]>:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let builders = [
        OpBuilder<(ins "Value":$src, "Type":$resultType)>
    ];
    
    let hasFolder = 1;
}

def Math_AddTensorScalarOp : Math_Op<"add_ts"> {
    let summary = "perform tensor-scalar add";
    let arguments = (ins RankedTensorOf<[AnyFloat, AnyInteger]> : $tensor,
                        AnyTypeOf<[AnyFloat, AnyInteger]> : $scalar);
    let results = (outs RankedTensorOf<[AnyFloat, AnyInteger]> : $res);
    let assemblyFormat = "operands attr-dict `:` type(operands) `->` type($res)";
}