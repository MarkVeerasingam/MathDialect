func.func @main(%arg0: i32, %arg1: i32) -> i32 {
  // Constant Folding: should become 30
  %c10 = math.constant 10 : i32
  %c20 = math.constant 20 : i32
  %const_sum = math.add %c10, %c20 : i32

  // Structural Folding: (a - b) + b should become just 'a' (%arg0)
  %diff = math.sub %arg0, %arg1 : i32
  %struct_sum = math.add %diff, %arg1 : i32

  // Identity Folding: x - x should become 0
  %zero = math.sub %struct_sum, %struct_sum : i32

  // Final result: 30 + %arg0 + 0
  %res1 = math.add %const_sum, %struct_sum : i32
  %res2 = math.add %res1, %zero : i32

  return %res2 : i32
}