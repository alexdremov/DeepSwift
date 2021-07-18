//
//  Supplementary.swift
//  
//
//  Created by  Alex Dremov on 16.07.2021.
//

import Foundation

extension Graph {
    public func relu() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                if $0 > 0 {
                    return $0
                } else {
                    return 0
                }
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                if $0 > 0 {
                    return 1
                } else {
                    return 0
                }
            }
        }, name: "ReLU")
    }
    
    public func exp() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.exp($0)
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                Foundation.exp($0)
            }
        }, name: "exp")
    }
    
    public func tanh() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.tanh($0)
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                let tanhRes = Foundation.tanh($0)
                return (1 - tanhRes * tanhRes)
            }
        }, name: "tanh")
    }
    
    public func sigmoid() -> FunctionNode {
        let sigm:(Matrix<MatrixDefType>) -> Matrix<MatrixDefType> = { inp in
            inp.map {
                1 / (1 + Foundation.exp(-$0))
            }
        }
        return FunctionNode(inp: self, shape: self.shape, functionForward: sigm,
                            functionBack: { frwd, grad in
                                    let sCalc = sigm(frwd)
                                    return ((1 - sCalc) * sCalc)
        }, name: "sigmoid")
    }
    
    public func lerelu(k: MatrixDefType = 0.01) -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                if $0 > 0 {
                    return $0
                } else {
                    return k * $0
                }
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                if $0 > 0 {
                    return 1
                } else {
                    return k
                }
            }
        }, name: "LeReLU")
    }
    
    public func elu(k: MatrixDefType = 0.01) -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                if $0 > 0 {
                    return $0
                } else {
                    return k * (Foundation.exp($0) - 1)
                }
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                if $0 > 0 {
                    return 1
                } else {
                    return k * (Foundation.exp($0))
                }
            }
        }, name: "ELU")
    }
    
    public func softmax() -> FunctionNode {
        FunctionNode(inp: self, shape: Shape(self.shape.row, 1), functionForward: {
            let maxEl = $0.max()
            let tmpProd = ($0 - maxEl).map{Foundation.exp($0)}
            return tmpProd / tmpProd.reduceSum()
        }, functionBack: { frwd, grad in
            grad
        }, name: "softmax")
    }
    
    public func log() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.log($0)
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                1 / $0
            }
        }, name: "log")
    }
    
    public func reduceSum(axis: Int = -1) -> FunctionNode {
        FunctionNode(inp: self, shape: Shape(1, 1), functionForward: { inp in
            inp.reduceSum(axis: axis)
        }, functionBack: { frwd, grad in
            grad
        }, name: "reduceSum")
    }
    
    public func sum() -> FunctionNode {
        FunctionNode(inp: self, shape: Shape(1, 1), functionForward: { inp in
            inp.sum()
        }, functionBack: { frwd, grad in
            Matrix.init(internalData: Array(repeating: grad[0, 0], count: self.shape.row * self.shape.col), dim: self.shape)
        }, name: "sum")
    }
    
    public func abs() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                if $0 > 0 {
                    return $0
                } else {
                    return -$0
                }
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                if $0 > 0 {
                    return 1
                } else {
                    return -1
                }
            }
        }, name: "abs")
    }
    
    public func pow(_ n: MatrixDefType) -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.pow($0, n)
            }
        }, functionBack: { frwd, grad in
            frwd.map {
                let a = Foundation.pow($0, n - 1) * n
                return a
            }
        }, name: "pow(\(n))")
    }
    
    var T: FunctionNode {
    FunctionNode(inp: self, shape: self.shape.T, functionForward: { inp in
            inp.T.as(MatrixDefType.self)
        }, functionBack: { frwd, grad in
            grad.T.as(MatrixDefType.self)
        }, name: "transpose")
    }
}
