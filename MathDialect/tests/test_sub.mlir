func.func @main(%arg0: i32) -> i32 {
  %c1 = arith.constant 5 : i32  
  %c2 = arith.constant 1 : i32
  %c3 = math.sub %c1, %c2 : i32
  %res = math.add %arg0, %c3 : i32
  return %res : i32
}