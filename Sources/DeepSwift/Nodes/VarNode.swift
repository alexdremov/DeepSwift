//
//  File.swift
//  
//
//  Created by  Alex Dremov on 13.07.2021.
//

import Foundation

public class VarNode<T: MatrixNumber>: Graph {
    public var context: GraphContext?

    public var frwd: Matrix<MatrixDefType>?

    public var children: [Graph] = []

    public var grad: Matrix<MatrixDefType>? = 0

    public var gradEnabled: Bool = true

    private(set) public var id: UUID = UUID()

    public var shape: Shape

    var name: String

    var value: Tensor<T>

    public var dumpDot: String {
        stringFriendlyID + "[label=\"Variable (\(name))\\n|{input:|output:}|{{[(–)]}|{[\(shape)]}}\"];\n"
    }

    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = value.as(MatrixDefType.self)
        return frwd!
    }

    public func _backward() throws {
//        grad = Matrix<MatrixDefType>.zero(dim: shape)
    }

    public init(_ value: Tensor<T>, name: String = "") {
        self.name = name
        if name == "" {
            self.name = String(describing: UUID())
        }
        self.value = value
        self.frwd = value.as(MatrixDefType.self)
        shape = value.shape
    }

    public func update(_ value: Tensor<T>) {
        if shape != value.shape {
            fatalError("Updated value must have the same shape")
        }
        self.value = value
    }
}

public typealias Input = VarNode
