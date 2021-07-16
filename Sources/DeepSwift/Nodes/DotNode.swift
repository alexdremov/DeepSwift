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
    
    public var grad: Matrix<MatrixDefType>? = nil
    
    public var gradEnabled: Bool = true
    
    public var shape: Shape
    
    public var id: UUID = UUID()

    var left, right: Graph
    
    public var dumpDot: String {
        ""
    }
    
    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = try left.forward().dot(right.forward())
        return frwd!
    }
    
    public func _backward() throws {
        left.grad = (left.grad ?? 0) + grad!.dot(right.frwd!.T)
        right.grad = (right.grad ?? 0) + left.frwd!.T.dot(grad!)
    }

    init(_ lhs: Graph, _ rhs: Graph){
        left = lhs
        right = rhs
        
        if !lhs.shape.dotable(shape: rhs.shape) {
            fatalError("Graph dimensions mismatch in dot: \(lhs.shape) and \(rhs.shape) ")
        }
        self.shape = lhs.shape.doted(rhs.shape)
        children = [lhs, rhs]
    }
    
    public func regrad() {
        grad = nil
        for i in children {
            i.regrad()
        }
    }
}
