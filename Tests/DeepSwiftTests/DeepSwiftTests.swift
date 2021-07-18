    import XCTest
    @testable import DeepSwift

    final class DeepSwiftTests: XCTestCase {
        func testMatrixBackprop() {
            let initWeight = Matrix<MatrixDefType>.random(dim: Shape(13, 7), generator: {MatrixDefType.random(in: -1.0...1.00)})
            let weightInput = Input(initWeight, name: "weight")
            
            let const:MatrixDefType = 23.0
            
            var graph = weightInput * ConstNode<MatrixDefType>(try! Matrix<MatrixDefType>(const).broadcast(shape: Shape(13, 7)))
            
            let res = try! graph.forward()
            
            try! graph.backward()
            
            XCTAssertEqual(res.data, initWeight.data.map{const * $0})
            XCTAssertEqual(weightInput.grad?.data, Array<MatrixDefType>(repeating: const, count: initWeight.shape.row * initWeight.shape.col))
        }
        
        func testsimpleOpt() {
            let x = Input(Matrix<MatrixDefType>(MatrixDefType.random(in: -1...1)), name: "x")
            
            var function = x * x + ConstNode<MatrixDefType>(12.0)
            let lambda = 0.1
            var loss = try! function.forward()
            for _ in 0..<200 {
                loss = try! function.forward()
                try! function.backward()
                x.update(x.value - lambda * x.grad!)
            }
            XCTAssertEqual(loss, 12.0)
        }
        
        func testsimpleOpt2D() {
            let x = Input(Matrix<MatrixDefType>.random(dim: Shape(5, 5), generator: {MatrixDefType.random(in: -1...1)}), name: "x")
            let y = Input(Matrix<MatrixDefType>.random(dim: Shape(5, 5), generator: {MatrixDefType.random(in: -1...1)}), name: "y")
            
            var function = ((x * y * ConstNode<MatrixDefType>(12.0) * y * x) + ConstNode<MatrixDefType>(12.0)).sum()
            let lambda = 0.01
            var loss = try! function.forward()
            for _ in 0..<500 {
                loss = try! function.forward()
                try! function.backward()
                x.update(x.value - lambda * x.grad!)
                y.update(y.value - lambda * y.grad!)
            }
            
            let res:Int = 12 * 25
            XCTAssertEqual(Int(loss[0, 0]), res)
        }
        
        
        func testsimpleMatrixOpt() {
            let x = Input<MatrixDefType>(Matrix.random(dim: Shape(5, 2), generator: {MatrixDefType.random(in: -10...10)}), name: "x")
            
            var function = (ConstNode<MatrixDefType>(7) * x * x + ConstNode<MatrixDefType>(12.0)).sum()
            let lambda = 0.1
            var loss = try! function.forward()
            for _ in 0..<100 {
                loss = try! function.forward()
                try! function.backward()
                x.update(x.value - lambda * x.grad!)
            }
            XCTAssertEqual(loss, 120)
        }
        
        func testsimpleMatrixOptMS() {
            let x = Input<MatrixDefType>(Matrix.random(dim: Shape(5, 2), generator: {MatrixDefType.random(in: -1...1)}), name: "x")
            let target = Input<MatrixDefType>(Matrix.zero(dim: Shape(5, 2)), name: "x")
            
            let function = (ConstNode<MatrixDefType>(7) * x * x + ConstNode<MatrixDefType>(12.0))
            
            var lossGraph = ((function - target).pow(2)).sum()
            
            let lambda = 0.001
            var loss = try! lossGraph.forward()
            for _ in 0..<1000 {
                loss = try! lossGraph.forward()
                try! lossGraph.backward()
                x.update(x.value - lambda * x.grad!)
            }
            XCTAssertEqual(loss, Matrix(12 * 12 * 10))
        }
        
        func testArithamcy() {
            let matrix = Matrix<MatrixDefType>.random(dim: Shape(1, 1), generator: {4})
            let x = Input(matrix, name: "x")
            
            var graph = x * x + x * 7
        
            XCTAssertEqual(try! graph.forward(), Matrix(4 * 4 + 7 * 4))
            try! graph.backward()
            XCTAssertEqual(x.grad!, Matrix(4 * 2 + 7))
        }
        
        func testPow() {
            let matrix = Matrix<MatrixDefType>.random(dim: Shape(5, 2), generator: {4})
            let xF = Input(matrix, name: "x")
            let xS = Input(matrix, name: "x")
            
            let functionFirst = xF * ConstNode<MatrixDefType>(7) + xF.pow(2) + xF * ConstNode<MatrixDefType>(7)
            
            print(functionFirst.dotFile)
            
            let functionSec = xS * ConstNode<MatrixDefType>(7) + xS * xS + xS * ConstNode<MatrixDefType>(7)
            
            var lossGraphFirst = functionFirst
            var lossGraphSec = functionSec
            
            XCTAssertEqual(try? lossGraphFirst.forward(), try? lossGraphSec.forward())
            
            try! lossGraphFirst.backward()
            try! lossGraphSec.backward()
            
            XCTAssertEqual(xF.grad, xS.grad)
        
        }
        
        func testPowOpt() {
            let x = Input<MatrixDefType>(Matrix.random(dim: Shape(5, 2), generator: {MatrixDefType.random(in: -1...1)}), name: "x")
            
            let y = Input<MatrixDefType>(Matrix.random(dim: Shape(5, 2), generator: {MatrixDefType.random(in: -1...1)}), name: "x")
            
            var lossGraph = (x * y * y + ConstNode<MatrixDefType>(6) * x).pow(2).sum()
            
            let lambda = 0.005
            var loss = try! lossGraph.forward()
            for _ in 0..<2000 {
                loss = try! lossGraph.forward()
                try! lossGraph.backward()
                x.update(x.value - lambda * x.grad!)
            }
            XCTAssertEqual(loss, Matrix(0))
        }
        
        func testLogOpt() {
            let variable = Input<MatrixDefType>(Matrix.random(dim: Shape(row: 5, col: 2),
                                                              generator: {MatrixDefType.random(in: -1...1)}), name: "x")
            
            let function = (variable.pow(2) + ConstNode<MatrixDefType>(1)).log().abs()
            var lossGraph = function.sum()
            
            let lambda = 0.01
            var loss = try! lossGraph.forward()
            for _ in 0..<5000 {
                loss = try! lossGraph.forward()
                try! lossGraph.backward()
                variable.update(variable.value - lambda * variable.grad!)
            }
            XCTAssertEqual(loss, Matrix(0))
        }
        
        func testNeuronModel() {
            
            func funcPredicted(x: MatrixDefType, y:MatrixDefType) -> Bool {
                y > 2 * x + 6
            }
            
            var (train, target) = { () -> ([[MatrixDefType]], [[MatrixDefType]]) in
                var train:[[MatrixDefType]] = []
                var targets:[[MatrixDefType]] = []
                
                for _ in 0..<10 {
                    let (x, y) = (MatrixDefType.random(in: -20...20), MatrixDefType.random(in: -20...20))
                    train.append([x, y])
                    targets.append([funcPredicted(x: x, y: y) ? 1.0: 0.0])
                }
                
                return (train, targets)
            }()
            
            var trainMatrix:[Matrix<MatrixDefType>] = []
            
            for row in train {
                trainMatrix.append(Matrix<MatrixDefType>(internalData: row, dim: Shape(1, row.count)))
            }
            
            let w = Input<MatrixDefType>(Matrix.random(dim: Shape(1, trainMatrix[0].shape.col), generator: {MatrixDefType.random(in: -1...1)}), name:"weight")
            
            let b = Input(Matrix<MatrixDefType>.random(dim: Shape(1, 1), generator: {MatrixDefType.random(in: -1...1)}), name:"bias")
        
            var neuron:Graph = ConstNode<Float>(0)
            
            for (i, ex) in zip(0..<trainMatrix.count, trainMatrix) {
                let neuronEx:Graph = ((ConstNode(ex) * w).sum() + b).sigmoid()
                let tragetEx:Graph = ConstNode<MatrixDefType>(Matrix(target[i][0]))
                
                let lossEx:Graph = tragetEx * neuronEx.log() +
                    (ConstNode<MatrixDefType>(1) - tragetEx) * (ConstNode<MatrixDefType>(1) - neuronEx).log()
                neuron = neuron - lossEx
            }
            
            neuron = neuron / ConstNode<Int>(Matrix(trainMatrix.count))
            
            print(neuron.dotFile)
            
            let lr = 0.001
            
            for epoch in 0..<1000 {
                let loss = try! neuron.forward()
                print("Epoch \(epoch): \(loss)")
                
                try! neuron.backward()
                
                w.update(w.value - lr * w.grad!)
                b.update(b.value - lr * b.grad!)
            }
            
            XCTAssertEqual(try! neuron.forward()[0,0], 0)
        }
    }
