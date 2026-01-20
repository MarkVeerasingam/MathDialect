include_directories(${MLIR_INCLUDE_DIRS})

# --- MathDialect ---
set(LLVM_TARGET_DEFINITIONS MathDialect.td)
# Generate Dialect Declarations and Definitions 
mlir_tablegen(MathDialect.h.inc -gen-dialect-decls -dialect=math)
mlir_tablegen(MathDialect.cpp.inc -gen-dialect-defs -dialect=math)
add_public_tablegen_target(MLIRMathDialectIncGen)

# --- MathOps ---
set(LLVM_TARGET_DEFINITIONS MathOps.td)
# Generate Ops Declarations and Definitions 
mlir_tablegen(MathOps.h.inc -gen-op-decls -dialect=math)
mlir_tablegen(MathOps.cpp.inc -gen-op-defs -dialect=math)
add_public_tablegen_target(MLIRMathOpsIncGen)