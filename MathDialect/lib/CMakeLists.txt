add_mlir_dialect_library(MLIRStandalone
        MathDialect.cpp

        DEPENDS
        MLIRMathDialectIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )