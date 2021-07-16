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
        }, functionBack: { inp in
            inp.map {
                if $0 > 0 {
                    return 1
                } else {
                    return 0
                }
            }
        })
    }
    
    public func exp() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.exp($0)
            }
        }, functionBack: { inp in
            inp.map {
                Foundation.exp($0)
            }
        })
    }
    
    public func tanh() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.tanh($0)
            }
        }, functionBack: { inp in
            inp.map {
                let tanhRes = Foundation.tanh($0)
                return 1 - tanhRes * tanhRes
            }
        })
    }
    
    public func sigmoid() -> FunctionNode {
        let sigm:(Matrix<MatrixDefType>) -> Matrix<MatrixDefType> = { inp in
            inp.map {
                1 / (1 + Foundation.exp(-$0))
            }
        }
        return FunctionNode(inp: self, shape: self.shape, functionForward: sigm, functionBack: { inp in
            let sCalc = sigm(inp)
            return (1 - sCalc) * sCalc
        })
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
        }, functionBack: { inp in
            inp.map {
                if $0 > 0 {
                    return 1
                } else {
                    return k
                }
            }
        })
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
        }, functionBack: { inp in
            inp.map {
                if $0 > 0 {
                    return 1
                } else {
                    return k * (Foundation.exp($0))
                }
            }
        })
    }
    
    public func softmax() -> FunctionNode {
        FunctionNode(inp: self, shape: Shape(self.shape.row, 1), functionForward: {
            let maxEl = $0.max()
            let tmpProd = ($0 - maxEl).map{Foundation.exp($0)}
            return tmpProd / tmpProd.reduceSum()
        }, functionBack: { inp in
            inp.map {
                $0
            }
        })
    }
    
    public func log() -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.log($0)
            }
        }, functionBack: { inp in
            inp.map {
                1 / $0
            }
        })
    }
    
    public func reduceSum(axis: Int = -1) -> FunctionNode {
        FunctionNode(inp: self, shape: Shape(1, 1), functionForward: { inp in
            inp.reduceSum(axis: axis)
        }, functionBack: { inp in
            inp
        })
    }
    
    public func sum() -> FunctionNode {
        FunctionNode(inp: self, shape: Shape(1, 1), functionForward: { inp in
            inp.sum()
        }, functionBack: { inp in
            inp
        })
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
        }, functionBack: { inp in
            inp.map {
                if $0 > 0 {
                    return 1
                } else {
                    return -1
                }
            }
        })
    }
    
    public func pow(_ n: MatrixDefType) -> FunctionNode {
        FunctionNode(inp: self, shape: self.shape, functionForward: { inp in
            inp.map {
                Foundation.pow($0, n)
            }
        }, functionBack: { inp in
            inp.map {
                Foundation.pow($0, n - 1) * n
            }
        })
    }
    
    var T: FunctionNode {
    FunctionNode(inp: self, shape: self.shape.T, functionForward: { inp in
            inp.T.as(MatrixDefType.self)
        }, functionBack: { inp in
            inp.T.as(MatrixDefType.self)
        })
    }
}
