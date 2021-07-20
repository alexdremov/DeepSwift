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

```swift
(Matrix<1, 5> -broadcasted-> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>

(Matrix<1, 1> -broadcasted-> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>

(Matrix<5, 1> -broadcasted-> Matrix<5, 5>) * Matrix<5, 5> = Matrix<5, 5>
```

### Element-wise multiplication (Hadamard product)


```swift
>>>>>>> 59e92956adc06c5edd582af59bba47f05761ddf6
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
=======
```swift
>>>>>>> 59e92956adc06c5edd582af59bba47f05761ddf6
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

### Computing gradient

Gradient is computed using backpropagation. Forward pass is required before executing backprop

```swift
let x = Input<Int>(Matrix(5), name:"Input variable")

var graph:Graph = x * x + 2 * x + 5
// Integer literals are transformed to ConstNodes

try? graph.forward()
try? graph.backward()

print(x.grad!.as(Int.self) == Matrix<Int>(2 * 5 + 2))
// d/dx(x^2 + 2 * x + 5) = 2 * x + 2
```

Partial derivatives are supported 

```swift
let x = Input<Int>(Matrix(5), name:"x")
let y = Input<Int>(Matrix(7), name:"y")

var graph:Graph = x * y + ConstNode<Int>(2) * (x + y) + ConstNode<Int>(5) * y
// Integer literals are transformed to ConstNodes

try? graph.forward()
try? graph.backward()

print(x.grad!.as(Int.self) == Matrix<Int>(7 + 2))
print(y.grad!.as(Int.self) == Matrix<Int>(5 + 2 + 5))
// d/dx(x * y + 2 * (x + y) + 5 * y) = y + 2
// d/dy(x * y + 2 * (x + y) + 5 * y) = x + 2 + 5
```
