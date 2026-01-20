add_mlir_dialect_library(MLIRToyMathDialect
        MathDialect.cpp
        MathOps.cpp

        DEPENDS
        MLIRMathDialectIncGen
        MLIRMathOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )