#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "MathDialect.h"

int main(int argc, char **argv)
{
    mlir::DialectRegistry registry;

    // Only register the dialects we are actually linking in CMake
    registry.insert<mlir::math::MathDialect,
                    mlir::arith::ArithDialect,
                    mlir::func::FuncDialect>();

    // Note: mlir::registerAllPasses() is removed to avoid undefined references
    // If you need specific passes (like -canonicalize), register them individually.
    mlir::registerCanonicalizerPass();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Math optimizer driver\n", registry));
}