//
// Created by Alex Dremov on 13.07.2021.
//

import Foundation

public protocol Graph {
    var id: UUID {get}

    var dumpDot: String {get}

    var shape: Shape {get}

    var children: [Graph] {get}

    var gradEnabled: Bool {get set}

    var context: GraphContext? {get set}

    var grad: Tensor<MatrixDefType>? {get set}

    var frwd: Matrix<MatrixDefType>? {get set}

    func forward() throws -> Tensor<MatrixDefType>

    func _backward() throws
}

public func +(lhs: Graph, rhs: Graph) -> AddNode {
    AddNode(lhs, rhs)
}

public func *(lhs: Graph, rhs: Graph) -> MulNode {
    MulNode(lhs, rhs)
}

public func -(lhs: Graph, rhs: Graph) -> AddNode {
    AddNode(lhs, ConstNode<Int>(-1) * rhs)
}

public func /(lhs: Graph, rhs: Graph) -> MulNode {
    MulNode(lhs, rhs ** ConstNode<Int>(-1))
}

public func **(lhs: Graph, rhs: Graph) -> PowNode {
    PowNode(lhs, rhs)
}

public func +<T: MatrixNumber>(lhs: Graph, rhs: T) -> AddNode {
    AddNode(lhs, ConstNode(Matrix(rhs)))
}

public func *<T: MatrixNumber>(lhs: Graph, rhs: T) -> MulNode {
    MulNode(lhs, ConstNode(Matrix(rhs)))
}

public func -<T: MatrixNumber>(lhs: Graph, rhs: T) -> AddNode {
    AddNode(lhs, ConstNode(Matrix(-1 * rhs)))
}

public func /<T: MatrixNumber>(lhs: Graph, rhs: T) -> MulNode {
    MulNode(lhs, ConstNode(Matrix(rhs)) ** ConstNode<Int>(-1))
}

public func **<T: MatrixNumber>(lhs: Graph, rhs: T) -> PowNode {
    PowNode(lhs, ConstNode(Matrix(rhs)))
}

public func +<T: MatrixNumber>(lhs: T, rhs: Graph) -> AddNode {
    AddNode(ConstNode(Matrix(lhs)), rhs)
}

public func *<T: MatrixNumber>(lhs: T, rhs: Graph) -> MulNode {
    MulNode(ConstNode(Matrix(lhs)), rhs)
}

public func -<T: MatrixNumber>(lhs: T, rhs: Graph) -> AddNode {
    AddNode(ConstNode(Matrix(lhs)), -1 * rhs)
}

public func /<T: MatrixNumber>(lhs: T, rhs: Graph) -> MulNode {
    MulNode(ConstNode(Matrix(lhs)), rhs ** ConstNode<Int>(-1))
}

public func **<T: MatrixNumber>(lhs: T, rhs: Graph) -> PowNode {
    PowNode(ConstNode(Matrix(lhs)), rhs)
}

extension Graph {
    mutating public func backward() throws {
        var topo: [Graph] = []
        var visited = Set<UUID>()

        func buildTopo(v: Graph) {
            if !visited.contains(v.id) {
                visited.insert(v.id)
                for child in v.children {
                    buildTopo(v: child)
                }
                topo.append(v)
            }
        }

        buildTopo(v: self)

        for v in 0..<topo.count {
            topo[v].grad = nil
        }

        grad = Matrix<MatrixDefType>.ones(dim: shape)
        for v in topo.reversed() {
            try v._backward()
        }
    }

//    public func dot(_ rhs: Graph) -> DotNode {
//        DotNode(self, rhs)
//    }

    public var _getDot: String {
        var res = dumpDot + "\n"
        for child in children {
            res += child.dumpDot + "\n"
        }
        return res
    }

    public var dotFile: String {
        let res = "digraph G {concentrate=True; rankdir=TB; node [shape=record];\n"
        return res + _getDot + "\n}"
    }

    public var stringFriendlyID: String {
        let exclude: Set<Character> = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        return String(String(describing: id).map {
            if $0 ==  "-" {
                return "_"
            }
            if exclude.contains($0) {
                return Character(UnicodeScalar($0.asciiValue! + Character("a").asciiValue! - Character("0").asciiValue!))
            }
            return $0
        })
    }
}
