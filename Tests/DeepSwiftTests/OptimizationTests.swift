//
//  File.swift
//  
//
//  Created by  Alex Dremov on 19.07.2021.
//

import XCTest
@testable import DeepSwift

final class OptimisationTests: XCTestCase {
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
        XCTAssertEqual(loss.sum()[0, 0], 12.0, accuracy: 1e-6)
    }

    func testsimpleOpt2D() {
        let x = Input(Matrix<MatrixDefType>.random(dim: Shape(5, 5), generator: {MatrixDefType.random(in: -1...1)}), name: "x")
        let y = Input(Matrix<MatrixDefType>.random(dim: Shape(5, 5), generator: {MatrixDefType.random(in: -1...1)}), name: "y")

        var function = ((x * y * ConstNode<MatrixDefType>(12.0) * y * x) + ConstNode<MatrixDefType>(12.0)).sum()
        let lambda = 0.01
        var loss = try! function.forward()
        for _ in 0..<1000 {
            loss = try! function.forward()
            try! function.backward()
            x.update(x.value - lambda * x.grad!)
            y.update(y.value - lambda * y.grad!)
        }

        let res: Double = 12 * 25
        XCTAssertEqual(Double(loss.sum()[0, 0]), res, accuracy: 1e-3)
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
        XCTAssertEqual(loss.sum()[0, 0], 120.0, accuracy: 1e-6)
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
        XCTAssertEqual(loss.sum()[0, 0], Matrix(12 * 12 * 10).sum()[0, 0], accuracy: 1e-6)
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
        XCTAssertEqual(loss.sum()[0, 0], 0, accuracy: 1e-6)
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
        XCTAssertEqual(loss.sum()[0, 0], 0, accuracy: 1e-6)
    }

    func testAbsOpt() {
        let variable = Input<MatrixDefType>(Matrix.random(dim: Shape(row: 5, col: 2),
                                                          generator: {MatrixDefType.random(in: -1...1)}), name: "x")

        let function = (variable.pow(3)).abs()
        var lossGraph = function.sum()

        let lambda = 0.1
        var loss = try! lossGraph.forward()
        for _ in 0..<5000 {
            loss = try! lossGraph.forward()
            try! lossGraph.backward()
            variable.update(variable.value - lambda * variable.grad!)
//                print(loss)
        }
        XCTAssertEqual(loss[0, 0], 0, accuracy: 1e-6)
    }

    func testNeuronModel() {

        func funcPredicted(x: MatrixDefType, y: MatrixDefType) -> Bool {
            y > 2 * x + 6
        }

        let (train, target) = { () -> ([[MatrixDefType]], [[MatrixDefType]]) in
            var train: [[MatrixDefType]] = []
            var targets: [[MatrixDefType]] = []

            for _ in 0..<5 {
                let (x, y) = (MatrixDefType.random(in: -20...20), MatrixDefType.random(in: -20...20))
                train.append([x, y])
                targets.append([funcPredicted(x: x, y: y) ? 1.0: 0.0])
            }

            return (train, targets)
        }()

//            print(train, target)

        var trainMatrix: [Matrix<MatrixDefType>] = []

        for row in train {
            trainMatrix.append(Matrix<MatrixDefType>(internalData: row, dim: Shape(1, row.count)))
        }

        let w = Input<MatrixDefType>(Matrix.random(dim: Shape(1, trainMatrix[0].shape.col),
                                                   generator: {0}), name: "weight")

        let b = Input(Matrix<MatrixDefType>.random(dim: Shape(1, 1), generator: {0}), name: "bias")

        var neuron: Graph = ConstNode<Float>(0)

        for (i, ex) in zip(0..<trainMatrix.count, trainMatrix) {
            let neuronEx: Graph = ((ConstNode(ex) * w).sum() + b).sigmoid()
            let tragetEx: Graph = ConstNode<MatrixDefType>(Matrix(target[i][0]))

            let lossEx: Graph = tragetEx * neuronEx.log() +
                (ConstNode<MatrixDefType>(1) - tragetEx) * (ConstNode<MatrixDefType>(1) - neuronEx).log()
            neuron = neuron - lossEx
        }

        neuron = neuron / ConstNode<Int>(Matrix(trainMatrix.count))

        let lr = 0.05

        for epoch in 0..<2000 {
            let loss = try! neuron.forward()
//            print("Epoch \(epoch): \(loss)")

            try! neuron.backward()

            w.update(w.value - lr * w.grad!)
            b.update(b.value - lr * b.grad!)
        }

        XCTAssertEqual(try! neuron.forward()[0, 0], 0, accuracy: 1e-2)
    }

    func testLinearApproximation() {

        func funcPredicted(x: MatrixDefType) -> MatrixDefType {
            0
        }

        let (train, test) = { () -> ([[MatrixDefType]], [[MatrixDefType]]) in
            var train: [[MatrixDefType]] = []
            var targets: [[MatrixDefType]] = []

            for _ in 0..<10 {
                let (x, _) = (MatrixDefType.random(in: -20...20), MatrixDefType.random(in: -20...20))
                train.append([x])
                targets.append([funcPredicted(x: x)])
            }

            return (train, targets)
        }()

//            print(train, target)

        let trainInp = Input<MatrixDefType>(Matrix(internalData: train))
        let testInp  = Input<MatrixDefType>(Matrix(internalData: test))

        let w = Input<MatrixDefType>(Matrix.random(dim: Shape(1, 1),
                                                   generator: {MatrixDefType.random(in: -1...1)}), name: "weight")

        let b = Input(Matrix<MatrixDefType>.random(dim: Shape(1, 1), generator: {MatrixDefType.random(in: -1...1)}), name: "bias")

        var neuron: Graph = (((trainInp * w).sum() + b) - testInp).pow(2).sumMean()

        let lr = 0.01

        for epoch in 0..<500 {
            let loss = try! neuron.forward()
//            print("Epoch \(epoch): \(loss)")

            try! neuron.backward()

            w.update(w.value - lr * w.grad!)
            b.update(b.value - lr * b.grad!)
        }

        XCTAssertEqual(try! neuron.forward()[0, 0], 0, accuracy: 1e-6)
    }

    func testReluOpt() {
        let x = Input<MatrixDefType>(Matrix.random(dim: Shape(1, 1),
                                                   generator: {MatrixDefType.random(in: 1..<2)}), name: "input")

        var neuron: Graph = x.relu()
        let lr = 0.1

        for _ in 0..<500 {
            let loss = try! neuron.forward()
            XCTAssertTrue((loss)[0, 0] >= 0)
            try! neuron.backward()

            x.update(x.value - x.grad! * lr)
        }

        XCTAssertEqual((try! neuron.forward())[0, 0], 0, accuracy: 1e-6)
    }

    func testReluComplexOpt() {
        let x = Input<MatrixDefType>(Matrix.random(dim: Shape(1, 1),
                                                   generator: {MatrixDefType.random(in: 1..<10)}), name: "input")

        var neuron: Graph = 3 * x.relu() + 2 * x.relu() + 3 * x.relu()
        let lr = 0.01

        for _ in 0..<5000 {
            let loss = try! neuron.forward()
            if x.value[0, 0] > 0 {
                XCTAssertTrue((loss)[0, 0] >= 0)
            }
            try! neuron.backward()
            if x.value[0, 0] > 0 {
                XCTAssertEqual(x.grad![0, 0], 8.0, accuracy: 1e-6)
            }

            x.update(x.value - x.grad! * lr)
        }

        XCTAssertEqual(try! neuron.forward()[0, 0], 0, accuracy: 1e-6)
    }
}
