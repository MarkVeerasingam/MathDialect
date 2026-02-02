func.func @test_splat() -> tensor<4xf32> {
  %cst = arith.constant 1.0 : f32
  %0 = math.splat %cst : f32 -> tensor<4xf32>
  return %0 : tensor<4xf32>
}