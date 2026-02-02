include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MathDialect.td"

// Base class
class Math_Op<string mnemonic, list<Trait> traits = []> :
    Op<Math_Dialect, mnemonic, traits>;

// Constant Op
def Math_ConstantOp : Math_Op<"constant", [Pure, ConstantLike]> {
    let summary = "Integer or floating-point constant";
    let description = [{
        The `math.constant` operation produces an SSA value equal to some constant
        specified by an attribute. This can be an integer or floating-point value.
        
        Example:
        %0 = math.constant 42 : i32
        %1 = math.constant 3.14 : f32
    }];
    
    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTypeOf<[AnyFloat, AnyInteger]>:$res);
    
    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;
    let hasVerifier = 1;
}

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
