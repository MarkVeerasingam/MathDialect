#include "MathOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::math;

// Verifier for ConstantOp
LogicalResult ConstantOp::verify()
{
    auto type = getType();
    auto valueAttr = getValue();

    // 1. Correctly cast to TypedAttr
    auto typedAttr = llvm::dyn_cast<mlir::TypedAttr>(valueAttr);

    if (!typedAttr)
    {
        return emitOpError("requires a typed attribute");
    }

    // 2. Call getType() on the CASTED variable (typedAttr), not valueAttr
    if (typedAttr.getType() != type)
    {
        return emitOpError("attribute type ")
               << typedAttr.getType() << " doesn't match return type " << type;
    }

    // 3. Verify it's either an integer or float attribute
    if (!llvm::isa<IntegerAttr, FloatAttr>(valueAttr))
    {
        return emitOpError("value must be an integer or float attribute");
    }

    return success();
}

// Folder for ConstantOp
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor)
{
    return getValue();
}

// Custom parser/printer stay the same as your code...
ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result)
{
    Attribute valueAttr;
    // This parses "5 : i32" as a single TypedAttr
    if (parser.parseAttribute(valueAttr))
        return failure();

    result.addAttribute("value", valueAttr);

    // Extract the type from the attribute to set the result type
    if (auto typedAttr = llvm::dyn_cast<TypedAttr>(valueAttr))
        result.addTypes(typedAttr.getType());

    return success();
}

void ConstantOp::print(OpAsmPrinter &p)
{
    p << " ";
    p.printAttribute(getValue()); // This prints "5 : i32"
}

// === Binary Op Folding Logic ===

// Template helper to handle Integer and Floating Point arithmetic
template <typename IntFn, typename FloatFn>
static OpFoldResult foldBinaryOp(Attribute lhs, Attribute rhs, IntFn intFn, FloatFn floatFn)
{
    if (!lhs || !rhs)
        return nullptr;

    if (auto l = llvm::dyn_cast<IntegerAttr>(lhs))
    {
        if (auto r = llvm::dyn_cast<IntegerAttr>(rhs))
        {
            return IntegerAttr::get(l.getType(), intFn(l.getValue(), r.getValue()));
        }
    }
    if (auto l = llvm::dyn_cast<FloatAttr>(lhs))
    {
        if (auto r = llvm::dyn_cast<FloatAttr>(rhs))
        {
            APFloat res = floatFn(l.getValue(), r.getValue());
            return FloatAttr::get(l.getType(), res);
        }
    }
    return nullptr;
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor)
{
    return foldBinaryOp(adaptor.getLhs(), adaptor.getRhs(), [](APInt a, APInt b)
                        { return a + b; }, [](APFloat a, APFloat b)
                        { a.add(b, APFloat::rmNearestTiesToEven); return a; });
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor)
{
    return foldBinaryOp(adaptor.getLhs(), adaptor.getRhs(), [](APInt a, APInt b)
                        { return a - b; }, [](APFloat a, APFloat b)
                        { a.subtract(b, APFloat::rmNearestTiesToEven); return a; });
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor)
{
    return foldBinaryOp(adaptor.getLhs(), adaptor.getRhs(), [](APInt a, APInt b)
                        { return a * b; }, [](APFloat a, APFloat b)
                        { a.multiply(b, APFloat::rmNearestTiesToEven); return a; });
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor)
{
    auto rhs = adaptor.getRhs();
    if (!rhs)
        return nullptr;

    // Check for Division by Zero
    if (auto r = llvm::dyn_cast<IntegerAttr>(rhs))
        if (r.getValue().isZero())
            return nullptr;
    if (auto r = llvm::dyn_cast<FloatAttr>(rhs))
        if (r.getValue().isZero())
            return nullptr;

    return foldBinaryOp(adaptor.getLhs(), rhs, [](APInt a, APInt b)
                        { return a.sdiv(b); }, [](APFloat a, APFloat b)
                        { a.divide(b, APFloat::rmNearestTiesToEven); return a; });
}

#define GET_OP_CLASSES
#include "MathOps.cpp.inc"