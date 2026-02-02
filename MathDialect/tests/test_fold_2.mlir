func.func @math_fold(%arg0: i32) -> i32 {
  %c0 = math.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = math.constant 10 : i32
  %c2 = arith.constant 2 : i32

  // Should fold to 20
  %0 = math.mul %c10, %c2 : i32
  // Should fold to 5
  %1 = math.div %0, %c2 : i32
  // Should fold to 0 (x * 0)
  %2 = math.mul %arg0, %c0 : i32
  // Should fold to %arg0 (x / 1)
  %3 = math.div %arg0, %c1 : i32
  
  %res = math.add %1, %3 : i32
  return %res : i32
}