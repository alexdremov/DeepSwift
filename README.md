# DeepSwift

Build dynamic computational graphs and compute gradients

## Examples

Simple SGD optimization:
Find minimum of sum_{elements} |log(1 + x ^ 2)|
```swift
import DeepSwift

let variable = Input<Float>(Matrix<Float>.random(dim: Shape(row: 5, col: 2), generator: {Float.random(in: -1...1)}), name: "x")
let function = (variable.pow(2) + ConstNode<Float>(1)).log().abs()
var lossGraph = function.sum()
            
let lr = 0.01
var loss = try! lossGraph.forward()
for _ in 0..<500 {
  loss = try! lossGraph.forward()
  print(loss)
  try! lossGraph.backward()
  
  variable.update(variable.value - lr * variable.grad!)
  
}

assert(loss == Matrix(0))
```
