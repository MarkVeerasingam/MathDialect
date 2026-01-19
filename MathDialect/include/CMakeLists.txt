include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS MathDialect.td)

# Generate Dialect Declarations and Definitions 
mlir_tablegen(MathDialect.h.inc -gen-dialect-decls -dialect=math)
mlir_tablegen(MathDialect.cpp.inc -gen-dialect-defs -dialect=math)

add_public_tablegen_target(MLIRMathDialectIncGen)