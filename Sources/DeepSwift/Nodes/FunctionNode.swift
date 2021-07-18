//
//  FunctionNode.swift
//  
//
//  Created by  Alex Dremov on 16.07.2021.
//

import Foundation

public class FunctionNode: Graph {
    typealias GraphFunction = (_ inp: Matrix<MatrixDefType>) -> Matrix<MatrixDefType>
    typealias GraphBackFunction = (_ frwd: Matrix<MatrixDefType>, _ grad: Matrix<MatrixDefType>) -> Matrix<MatrixDefType>
    public var id = UUID()
    
    public var dumpDot: String {
        stringFriendlyID + "[label=\"Function: \(name)\\n|{input:|output:}|{{[\(children[0].shape)]}|{[\(shape)]}}\"];\n" +
            children[0].stringFriendlyID + " -> " + stringFriendlyID + "\n" + "\n" + children[0].dumpDot
    }
    
    public var shape: Shape
    
    private(set) public var name: String = "custom"
    
    public var children: [Graph]
    
    public var gradEnabled: Bool = true
    
    public var context: GraphContext?
    
    public var grad: Matrix<MatrixDefType>?
    
    public var frwd: Matrix<MatrixDefType>?
    
    var functionForward: GraphFunction
    
    var functionBack: GraphBackFunction
    
    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = try functionForward(children[0].forward())
        return frwd!
    }
    
    public func _backward() throws {
        children[0].grad = (children[0].grad ?? 0) + functionBack(children[0].frwd!, grad!)
    }
    
    init(inp: Graph, shape inShape: Shape,
         functionForward fForward: @escaping GraphFunction,
         functionBack fBack: @escaping GraphBackFunction, name: String="custom") {
        children = [inp]
        shape = inShape
        functionForward = fForward
        functionBack = fBack
        self.name = name
    }
}
