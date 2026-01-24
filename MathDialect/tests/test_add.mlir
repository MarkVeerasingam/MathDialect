func.func @main(%arg0: i32) -> i32 {
  %c1 = math.constant 5 : i32
  %res = math.add %arg0, %c1 : i32
  return %res : i32
}