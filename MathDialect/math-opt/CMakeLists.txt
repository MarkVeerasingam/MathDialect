set(LIBS
  MLIRToyMathDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIROptLib
  MLIRIR
)

add_llvm_executable(math-opt math-opt.cpp)
llvm_update_compile_flags(math-opt)
target_link_libraries(math-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(math-opt)