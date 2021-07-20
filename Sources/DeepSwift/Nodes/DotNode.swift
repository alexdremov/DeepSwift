//
//  DotNode.swift
//  
//
//  Created by  Alex Dremov on 16.07.2021.
//

import Foundation

import Foundation

public class DotNode: Graph {
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
        stringFriendlyID + "[label=\"Dot\\n|{input:|output:}|{{[(\(left.shape), \(right.shape))]}|{[\(shape)]}}\"];\n" +
            left.stringFriendlyID + " -> " + stringFriendlyID + "\n" +
            right.stringFriendlyID + " -> " + stringFriendlyID + "\n" + left.dumpDot + right.dumpDot
    }

    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = try left.forward().dot(right.forward())
        return frwd!
    }

    public func _backward() throws {

        left.grad = (left.grad ?? 0) + grad!.dot(right.frwd!.T)
        right.grad = (right.grad ?? 0) + grad!.T.dot(left.frwd!).T

        assert(left.grad?.shape == left.shape, "Shapes mismatch: \(String(describing: left.grad?.shape)) and \(left.shape)")
        assert(right.grad?.shape == right.shape, "Shapes mismatch: \(String(describing: right.grad?.shape)) and \(right.shape)")
    }

    init(_ lhs: Graph, _ rhs: Graph) {
        if !lhs.shape.dotable(shape: rhs.shape) {
            fatalError("Graph dimensions mismatch in dot: \(lhs.shape) and \(rhs.shape) ")
        }
        self.shape = lhs.shape.doted(rhs.shape)
        children = [lhs, rhs]
    }
}
