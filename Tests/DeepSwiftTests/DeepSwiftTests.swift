    import XCTest
    @testable import DeepSwift

    final class DeepSwiftTests: XCTestCase {
        
        func testMatrixBackprop() {
            let initWeight = Matrix<Float>.random(dim: Shape(13, 7), generator: {Float.random(in: -1.0...1.00)})
            let weightInput = Input(initWeight, name: "weight")
            
            let const:Float = 23.0
            
            var graph = weightInput * ConstNode<Float>(Matrix<Float>(const))
            
            let res = try! graph.forward()
            
            try! graph.backward()
            
            XCTAssertEqual(res.data, initWeight.data.map{const * $0})
            XCTAssertEqual(weightInput.grad?.data, Array<MatrixDefType>(repeating: const, count: initWeight.shape.row * initWeight.shape.col))
        }
        
        func testMatrixBackpropDotSimple() {
            let initWeight = Matrix<Float>.random(dim: Shape(1, 1), generator: {Float.random(in: -1.0...1.00)})
            let weightInput = Input(initWeight, name: "weight")
            
            let const:Float = 23.0
            
            var graph = ConstNode<Float>(Matrix<Float>(const)).dot(weightInput)
            
            let res = try! graph.forward()
            
            try! graph.backward()
            
            XCTAssertEqual(res.data, initWeight.data.map{const * $0})
            XCTAssertEqual(weightInput.grad?.data, Array<MatrixDefType>(repeating: const, count: 1))
        }
        
        func testMatrixBackpropDot() {
            let initWeight = Matrix<Float>.random(dim: Shape(7, 13), generator: {Float.random(in: -1.0...1.00)})
            let weightInput = Input(initWeight, name: "weight")
            
            let const:Float = 23.0
            
            let c = ConstNode<Float>(Matrix<Float>.random(dim: Shape(13, 7), generator: {Float.random(in: -1.0...1.00)}))
            
            var graph = c.dot(weightInput)
            
            let res = try! graph.forward()
            
            try! graph.backward()
            
            XCTAssertEqual(weightInput.grad!.T, c.value)
        }
        
        func testMatrixBackpropDotSecond() {
            let initWeight = Matrix<Float>.random(dim: Shape(13, 7), generator: {Float.random(in: -1.0...1.00)})
            let weightInput = Input(initWeight, name: "weight")
            
            let const:Float = 23.0
            
            let c = ConstNode<Float>(Matrix<Float>.random(dim: Shape(7, 13), generator: {Float.random(in: -1.0...1.00)}))
            
            var graph = weightInput.dot(c)
            
            let res = try! graph.forward()
            
            try! graph.backward()
            
            XCTAssertEqual(weightInput.grad!.T, c.value)
        }
        
        func testNeuron() {
            let dataset:Matrix<Double>, targets:Matrix<Double>
            (dataset, targets) = {
                var res:[[Int]] = []
                var map:[[Int]] = []
                
                for i in -100..<100 {
                    let randomed = Int.random(in: -10...10)
                    res.append([i, randomed])
                    map.append([(randomed > 5 + 2 * i) ? 1: 0])
                }
                
                return (Matrix<Int>(internalData: res).as(Double.self), Matrix<Int>(internalData: map).as(Double.self))
            }()
            
            let dataInput = Input(dataset.T, name: "data")
            let initWeight = Matrix<Float>.random(dim: Shape(dataset.shape.col, 1), generator: {Float.random(in: -1.0...1.00)})
            let weightInput = Input(initWeight, name: "weight")
            
            let initBias = Matrix<Float>.random(dim: Shape(1, 1), generator: {Float.random(in: -1.0...1.00)})
            let biasInput = Input(initBias, name: "bias")
            
            let neuron = (dataInput.T.dot(weightInput) + biasInput).sigmoid()
            
            let targetsNode = ConstNode(targets)
            
            var neuronAndLoss = (targetsNode - neuron).sum()
//            
//            print("Weight: ", weightInput.grad!, weightInput.value)
//            print("Bias: ", biasInput.grad!.reduceSum(axis: 1) / biasInput.grad!.shape.row, biasInput.value)
            let lambda = -0.1
            for _ in 0..<500 {
                let loss = try! neuronAndLoss.forward()
                try! neuronAndLoss.backward()
                weightInput.update(weightInput.value - lambda * weightInput.grad!)
                biasInput.update(biasInput.value - lambda * biasInput.grad!.sum() / biasInput.grad!.shape.row)
                print(loss)
            }
        }
        
        func testsimpleOpt() {
            let x = Input<Float>(Matrix<Float>(Float.random(in: -10...10)), name: "x")
            
            var function = x * x + ConstNode<Float>(12.0)
            let lambda = 0.1
            var loss = try! function.forward()
            for _ in 0..<100 {
                loss = try! function.forward()
                try! function.backward()
                x.update(x.value - lambda * x.grad!)
            }
            XCTAssertEqual(loss, 12.0)
        }
        
        
        func testsimpleMatrixOpt() {
            let x = Input<Float>(Matrix<Float>.random(dim: Shape(5, 2), generator: {Float.random(in: -10...10)}), name: "x")
            
            var function = (ConstNode<Float>(7) * x * x + ConstNode<Float>(12.0)).sum()
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
            let x = Input<Float>(Matrix<Float>.random(dim: Shape(5, 2), generator: {Float.random(in: -1...1)}), name: "x")
            let target = Input<Float>(Matrix<Float>.zero(dim: Shape(5, 2)), name: "x")
            
            let function = (ConstNode<Float>(7) * x * x + ConstNode<Float>(12.0))
            
            var lossGraph = ((function - target)).sum()
            
            let lambda = 0.1
            var loss = try! lossGraph.forward()
            for _ in 0..<100 {
                loss = try! lossGraph.forward()
                try! lossGraph.backward()
                x.update(x.value - lambda * x.grad!)
                print(loss)
            }
            XCTAssertEqual(loss, 120)
        }
    }
