//
// Created by Alex Dremov on 13.07.2021.
//

import Foundation

public protocol Graph {
    var id: UUID {get}

    var dumpDot: String {get}
    
    var shape: Shape {get}
    
    var children: [Graph] {get}
    
    var gradEnabled:Bool {get set}
    
    var context:GraphContext? {get set}
    
    func regrad()
    
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

extension Graph {
    mutating public func backward() throws {
        var topo:[Graph] = []
        var visited = Set<UUID>()
        
        regrad()
        
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
        grad = Matrix<MatrixDefType>.zero(dim: shape).map{_ in 1}
        for v in topo.reversed() {
            try v._backward()
        }
    }
    
    public func dot(_ rhs: Graph) -> DotNode {
        DotNode(self, rhs)
    }
}


