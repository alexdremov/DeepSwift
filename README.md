# DeepSwift

Build dynamic computational graphs and compute gradients

## Examples

Simple SGD optimization:
```swift
import DeepSwift

let variable = Input<Float>(Matrix<Float>.random(dim: Shape(row: 5, col: 2), generator: {Float.random(in: -1...1)}), name: "x")
let function = (variable.pow(2) + ConstNode<Float>(1)).log().abs()
var lossGraph = function.sum()
            
let lambda = 0.01
var loss = try! lossGraph.forward()
for _ in 0..<500 {
  loss = try! lossGraph.forward()
  try! lossGraph.backward()
  variable.update(variable.value - lambda * variable.grad!)
}

assert(loss == Matrix(0))
```
