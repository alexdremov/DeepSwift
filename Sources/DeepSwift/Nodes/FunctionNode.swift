//
//  FunctionNode.swift
//  
//
//  Created by  Alex Dremov on 16.07.2021.
//

import Foundation

public class FunctionNode: Graph {
    typealias GraphFunction = (_ inp: Matrix<MatrixDefType>) -> Matrix<MatrixDefType>
    public var id = UUID()
    
    public var dumpDot: String {
        ""
    }
    
    public var shape: Shape
    
    public var children: [Graph]
    
    public var gradEnabled: Bool = true
    
    public var context: GraphContext?
    
    public var grad: Matrix<MatrixDefType>?
    
    public var frwd: Matrix<MatrixDefType>?
    
    var functionForward: GraphFunction
    
    var functionBack: GraphFunction
    
    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = try functionForward(children[0].forward())
        return frwd!
    }
    
    public func _backward() throws {
        children[0].grad = functionBack(grad!)
    }
    
    init(inp: Graph, shape inShape: Shape,
         functionForward fForward: @escaping GraphFunction,
         functionBack fBack: @escaping GraphFunction) {
        children = [inp]
        shape = inShape
        functionForward = fForward
        functionBack = fBack
    }
    
    public func regrad() {
        grad = nil
        for i in children {
            i.regrad()
        }
    }
}
