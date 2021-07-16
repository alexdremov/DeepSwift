//
// Created by  Alex Dremov on 13.07.2021.
//

import Foundation

public class AddNode: Graph {
    public var context: GraphContext?
    
    public var frwd: Matrix<MatrixDefType>?
    
    public var children: [Graph]
    
    public var grad: Matrix<MatrixDefType>? = nil
    
    public var gradEnabled: Bool = true
    
    public var shape: Shape
    
    public var id:UUID = UUID()
    
    var left, right: Graph

    public var dumpDot: String {
        ""
    }
    
    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = try left.forward() + right.forward()
        return frwd!
    }
    
    public func _backward() throws {
        left.grad = (left.grad ?? 0) + grad!
        right.grad = (right.grad ?? 0) + grad!
    }

    init(_ lhs: Graph, _ rhs: Graph){
        left = lhs
        right = rhs
        
        if !lhs.shape.broadcastable(shape: rhs.shape) {
            fatalError("Graph dimensions mismatch in addition: \(lhs.shape) and \(rhs.shape) ")
        }
        shape = Shape.broadcasted(lhs.shape, rhs.shape)
        children = [lhs, rhs]
    }
    
    public func regrad() {
        grad = nil
        for i in children {
            i.regrad()
        }
    }
}
