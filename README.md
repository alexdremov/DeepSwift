# DeepSwift

Build dynamic computational graph with forward and backward propagation functionality

## Example

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
## Matrix operations

### Bradcasting
Element-wise operations support broadcasting similarly to numpy

(Matrix<1, 5> - broadcasted -> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>

(Matrix<1, 1> - broadcasted -> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>

(Matrix<5, 1> - broadcasted -> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>
