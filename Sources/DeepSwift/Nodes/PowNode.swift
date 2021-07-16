//
//  File.swift
//  
//
//  Created by  Alex Dremov on 15.07.2021.
//

import Foundation

public class PowNode: Graph {
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
        frwd = try left.forward() ** right.forward()[0, 0]
        return frwd!
    }
    
    public func _backward() throws {
        // todo: u ** v
        left.grad = (left.grad ?? 0) + grad! * (right.frwd! * left.frwd! ** (right.frwd! - 1))
        right.grad = 0
    }

    init(_ lhs: Graph, _ rhs: Graph){
        left = lhs
        right = rhs
        
        if !lhs.shape.broadcastable(shape: rhs.shape) {
            fatalError("Graph dimensions mismatch in power: \(lhs.shape) and \(rhs.shape) ")
        }
        self.shape = Shape.broadcasted(lhs.shape, rhs.shape)
        children = [lhs, rhs]
    }
    
    public func regrad() {
        grad = nil
        for i in children {
            i.regrad()
        }
    }
}
