//
// Created by Alex Dremov on 13.07.2021.
//

import Foundation

public class DivNode: Graph {
    public var context: GraphContext?

    public var frwd: Matrix<MatrixDefType>?

    public var children: [Graph]

    public var grad: Matrix<MatrixDefType>?

    public var gradEnabled: Bool = true

    public var shape: Shape

    public var id: UUID = UUID()

    var left: Graph {
        get {
            children[0]
        }
        set {
            children[0] = newValue
        }
    }

    var right: Graph {
        get {
            children[1]
        }
        set {
            children[1] = newValue
        }
    }

    public var dumpDot: String {
        stringFriendlyID + "[label=\"Div\\n|{input:|output:}|{{[(\(left.shape), \(right.shape))]}|{[\(shape)]}}\"];\n" +
            left.stringFriendlyID + " -> " + stringFriendlyID + "\n" +
            right.stringFriendlyID + " -> " + stringFriendlyID + "\n" + left.dumpDot + right.dumpDot
    }

    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = try left.forward() / right.forward()
        return frwd!
    }

    public func _backward() throws {
        left.grad = try ((left.grad ?? Matrix<MatrixDefType>.zero(dim: left.shape)) + grad! * (1 / right.frwd!))
            .reduceMean(shape: left.shape)
        right.grad = try ((right.grad ?? Matrix<MatrixDefType>.zero(dim: right.shape)) - grad! * left.frwd! * (1 / (right.frwd! * right.frwd!)))
            .reduceMean(shape: right.shape)

        assert(left.grad?.shape == left.shape, "Shapes mismatch: \(String(describing: left.grad?.shape)) and \(left.shape)")
        assert(right.grad?.shape == right.shape, "Shapes mismatch: \(String(describing: right.grad?.shape)) and \(right.shape)")
    }

    init(_ lhs: Graph, _ rhs: Graph) {
        if !lhs.shape.broadcastable(shape: rhs.shape) {
            fatalError("Graph dimensions mismatch in division: \(lhs.shape) and \(rhs.shape) ")
        }
        self.shape = Shape.broadcasted(lhs.shape, rhs.shape)
        children = [lhs, rhs]
    }
}
