//
// Created by Alex Dremov on 13.07.2021.
//

import Foundation

class ConstNode<T: MatrixNumber>: Graph & ExpressibleByIntegerLiteral  & ExpressibleByFloatLiteral {
    typealias IntegerLiteralType = Int
    typealias FloatLiteralType = MatrixDefType

    var context: GraphContext?

    var frwd: Matrix<MatrixDefType>?

    var children: [Graph] = []

    var grad: Matrix<MatrixDefType>?

    var gradEnabled: Bool = false

    var shape: Shape

    var id: UUID = UUID()

    var value: Tensor<T>

    var dumpDot: String {
        stringFriendlyID + "[label=\"Constant\\n|{input:|output:}|{{[(–)]}|{[\(shape)}}\"];\n"
    }

    public func forward() throws -> Matrix<MatrixDefType> {
        frwd = value.as(MatrixDefType.self)
        return frwd!
    }

    public func _backward() throws {
        grad = Matrix<MatrixDefType>.zero(dim: shape)
    }

    public init(_ val: Tensor<T>) {
        value = val
        shape = val.shape
        frwd = val.as(MatrixDefType.self)
    }

    required init(integerLiteral value: Int) {
        self.value = Matrix<T>(T(convertible: value))
        shape = self.value.shape
        frwd = self.value.as(MatrixDefType.self)
    }

    required init(floatLiteral value: MatrixDefType) {
        self.value = Matrix<T>(T(convertible: value))
        shape = self.value.shape
        frwd = self.value.as(MatrixDefType.self)
    }
}
