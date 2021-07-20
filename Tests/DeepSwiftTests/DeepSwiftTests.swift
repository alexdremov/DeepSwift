import XCTest
@testable import DeepSwift

final class DeepSwiftTests: XCTestCase {
    func testMatrixBackprop() {
        let initWeight = Matrix<MatrixDefType>.random(dim: Shape(13, 7), generator: {MatrixDefType.random(in: -1.0...1.00)})
        let weightInput = Input(initWeight, name: "weight")
        
        let const: MatrixDefType = 23.0
        
        var graph = weightInput * ConstNode<MatrixDefType>(try! Matrix<MatrixDefType>(const).broadcast(shape: Shape(13, 7)))
        
        let res = try! graph.forward()
        
        try! graph.backward()
        
        XCTAssertEqual(res.data, initWeight.data.map {const * $0})
        XCTAssertEqual(weightInput.grad?.data, [MatrixDefType](repeating: const, count: initWeight.shape.row * initWeight.shape.col))
    }
    
    func testArithamcy() {
        let matrix = Matrix<MatrixDefType>.random(dim: Shape(1, 1), generator: {4})
        let x = Input(matrix, name: "x")
        
        var graph = x * x + x * 7
        
        XCTAssertEqual(try! graph.forward()[0, 0], Matrix(4 * 4 + 7 * 4)[0, 0], accuracy: 1e-6)
        try! graph.backward()
        XCTAssertEqual(x.grad![0, 0], 4 * 2 + 7, accuracy: 1e-6)
    }
    
    func testPow() {
        let matrix = Matrix<MatrixDefType>.random(dim: Shape(5, 2), generator: {4})
        let xF = Input(matrix, name: "x")
        let xS = Input(matrix, name: "x")
        
        let functionFirst = xF * ConstNode<MatrixDefType>(7) + xF.pow(2) + xF * ConstNode<MatrixDefType>(7)
        
        //            print(functionFirst.dotFile)
        
        let functionSec = xS * ConstNode<MatrixDefType>(7) + xS * xS + xS * ConstNode<MatrixDefType>(7)
        
        var lossGraphFirst = functionFirst
        var lossGraphSec = functionSec
        
        XCTAssertEqual(try! lossGraphFirst.forward().sum()[0, 0], try! lossGraphSec.forward().sum()[0, 0], accuracy: 1e-6)
        
        try! lossGraphFirst.backward()
        try! lossGraphSec.backward()
        
        XCTAssertEqual(xF.grad!.sum()[0, 0], xS.grad!.sum()[0, 0], accuracy: 1e-6)
        
    }
    
    func testSimpleSum() {
        let koef = ConstNode<MatrixDefType>(Matrix.random(dim: Shape(row: 5, col: 2),
                                                          generator: {MatrixDefType.random(in: -1...1)}))
        let variable = Input<MatrixDefType>(Matrix.random(dim: Shape(row: 5, col: 2),
                                                          generator: {MatrixDefType.random(in: -1...1)}), name: "x")
        
        var lossGraph = (koef * variable).sum()
        
        _ = try! lossGraph.forward()
        try! lossGraph.backward()
        
        XCTAssertEqual(variable.grad!.sum()[0, 0], koef.value.sum()[0, 0], accuracy: 1e-6)
    }
    
    func testSum() {
        var koef = ConstNode<MatrixDefType>(Matrix.random(dim: Shape(row: 1, col: 1),
                                                          generator: {MatrixDefType.random(in: -1...1)}))
        var variable = Input<MatrixDefType>(Matrix.random(dim: Shape(row: 1, col: 1),
                                                          generator: {MatrixDefType.random(in: -1...1)}), name: "x")
        
        var lossGraph = 7 * variable + (koef * variable).sum() + 7 * variable
        
        _ = try! lossGraph.forward()
        try! lossGraph.backward()
        
        XCTAssertTrue(abs(variable.grad![0, 0] - (koef.value + 7 * 2)[0, 0]) < 1e-6)
        
        koef = ConstNode<MatrixDefType>(Matrix.random(dim: Shape(row: 2, col: 1),
                                                      generator: {MatrixDefType.random(in: -1...1)}))
        variable = Input<MatrixDefType>(Matrix.random(dim: Shape(row: 2, col: 1),
                                                      generator: {MatrixDefType.random(in: -1...1)}), name: "x")
        
        lossGraph = 7 * variable + (koef * variable).sum() + 7 * variable
        
        _ = try! lossGraph.forward()
        try! lossGraph.backward()
        
        XCTAssertEqual(variable.grad!.sum()[0, 0], (koef.value + 7 * 2).sum()[0, 0], accuracy: 1e-6)
    }
    
    func testAbs() {
        let variable = Input<MatrixDefType>(Matrix.random(dim: Shape(row: 1, col: 1),
                                                          generator: {MatrixDefType.random(in: 0.1...5)}), name: "x")
        
        var function = 7 * variable + (variable.pow(3)).abs() + 7 * variable
        _ = try! function.forward()
        try! function.backward()
        
        XCTAssertEqual(variable.grad![0, 0], 7 * 2 + (3 * variable.value * variable.value)[0, 0], accuracy: 1e-6)
    }
    
    func testSimpleGradient() {
        let x = Input<Int>(Matrix(5), name: "Input variable")
        
        var graph: Graph = x * x + 2 * x + 5
        // Integer literals are transformed to ConstNodes
        
        _ = try? graph.forward()
        try? graph.backward()
        
        XCTAssertTrue(x.grad!.as(Int.self) == Matrix<Int>(2 * 5 + 2))
    }
    
    func testSimplePartial() {
        let x = Input<Int>(Matrix(5), name: "x")
        let y = Input<Int>(Matrix(7), name: "y")
        
        var graph: Graph = x * y + ConstNode<Int>(2) * (x + y) + ConstNode<Int>(5) * y
        // Integer literals are transformed to ConstNodes
        
        _ = try? graph.forward()
        try? graph.backward()
        
        XCTAssertTrue(x.grad!.as(Int.self) == Matrix<Int>(7 + 2))
        XCTAssertTrue(y.grad!.as(Int.self) == Matrix<Int>(5 + 2 + 5))
    }
}
