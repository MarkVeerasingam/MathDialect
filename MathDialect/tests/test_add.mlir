module {
  func.func @add(%arg0: i32) -> i32 {
    // case 1: Constant Folding (10 + 20)... Should become 30
    %c10 = arith.constant 10 : i32
    %c20 = arith.constant 20 : i32
    %0 = math.add %c10, %c20 : i32

    // case 2: Identity Folding (arg0 + 0)... Should become %arg0
    %c0 = arith.constant 0 : i32
    %1 = math.add %arg0, %c0 : i32

    %2 = math.add %0, %1 : i32
    return %2 : i32
  }
}