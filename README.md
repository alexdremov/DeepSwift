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

```
(Matrix<1, 5> -broadcasted-> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>

(Matrix<1, 1> -broadcasted-> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>

(Matrix<5, 1> -broadcasted-> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>
```

### Element-wise multiplication (Hadamard product)

```
let a: Matrix<Int> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
]

let b: Matrix<Int> = [
            [-1, 2, -3],
            [4, -5, 6],
            [-7, 8, -9]
]

a * b == Matrix(internalData: [
                [1 * -1, 2 * 2 , 3 * -3],
                [4 * 4 , 5 * -5, 6 * 6 ],
                [7 * -7, 8 * 8 , 9 * -9],
            ])

```

In the same manner supported:
- Addition
- Substraction
- Division

## Graph

Computational graph has several restrictions:
- Scalar values are 1x1 matrices
- N-dimensional tensors are not supported
- Must be directed & acyclic. You need to make sure that there is no cycles. In case of cyclic graph, forward and backward propagations will run infinitely.

###  Building elements

Graph consists of several types of elements: variables – `Input()`, constants – `ConstNode()`, and operations – `+, -, /, *, **, functions`.

Simple example:

```
let x = Input<Int>(Matrix(5), name:"Input variable")

var graph:Graph = x * x + 2 * x + 5
// Integer literals are transformed to ConstNodes

print(try? graph.forward().as(Int.self) == Matrix<Int>(5 * 5 + 2 * 5 + 5))

x.update(0)

print(try? graph.forward().as(Int.self) == Matrix<Int>(0 * 0 + 2 * 0 + 5))

```

###  Functions

Almost all needed function are implemented. The interface for introducig new functions is provoded.

Supported element-wise functions:

- tanh
- sigmoid
- abs
- pow
- sum
- reduceMean
- reduceSum
- log
- ReLU
- ELU
- LeReLU
- exp
