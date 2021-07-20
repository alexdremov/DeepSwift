//
//  Matrix.swift
//  
//
//  Created by  Alex Dremov on 13.07.2021.
//

import Foundation
import Accelerate

public typealias Tensor = Matrix
public typealias MatrixDefType = Double
public typealias MatrixNumber = FloatConvertible & AdditiveArithmetic & Numeric & Comparable
infix operator **: MultiplicationPrecedence

public struct Shape: Equatable & ExpressibleByArrayLiteral & CustomStringConvertible {
    public typealias ArrayLiteralElement = Int

    public var row: Int, col: Int

    var T: Shape {
        Shape(row: col, col: row)
    }

    var shape: (row: Int, col: Int) {
        (row: row, col: col)
    }

    public var description: String {
        "(\(row), \(col))"
    }

    init (row: Int, col: Int) {
        self.row = row
        self.col = col
    }

    init (_ row: Int, _ col: Int) {
        self.row = row
        self.col = col
    }

    init (_ dim: (Int, Int)) {
        (self.row, self.col) = dim
    }

    public func broadcastable(shape: Shape) -> Bool {
        shape == (1, 1) || self.shape == (1, 1) ||
            (shape.row == self.shape.row && (shape.col == 1 || self.shape.col == 1)) ||
            (shape.col == self.shape.col && (shape.row == 1 || self.shape.row == 1)) ||
            self.shape == shape
    }

    public init(arrayLiteral elements: Int...) {
        assert(elements.count == 2, "Must be 2 dimensional")
        row = elements[0]
        col = elements[0]
    }

    public static func ==(lhs: Shape, rhs: (Int, Int)) -> Bool {
        lhs.col == rhs.1 && lhs.row == rhs.0
    }

    public static func ==(rhs: (Int, Int), lhs: Shape) -> Bool {
        lhs.col == rhs.1 && lhs.row == rhs.0
    }

    public static func ==(lhs: Shape, rhs: Shape) -> Bool {
        lhs.col == rhs.col && lhs.row == rhs.row
    }

    public static func broadcasted(_ lhs: Shape, _ rhs: Shape) -> Shape {
        if lhs == rhs {
            return lhs
        }
        if lhs == (1, 1) {
            return rhs
        }
        if rhs == (1, 1) {
            return lhs
        }
        if lhs.row == rhs.row {
            if lhs.col == 1 {
                return Shape(row: lhs.row, col: rhs.col)
            }
            if rhs.col == 1 {
                return Shape(row: lhs.row, col: lhs.col)
            }
        }
        if lhs.col == rhs.col {
            if lhs.row == 1 {
                return Shape(row: rhs.row, col: rhs.col)
            }
            if rhs.row == 1 {
                return Shape(row: lhs.row, col: rhs.col)
            }
        }
        fatalError("Cannot broadcast \(lhs) and \(rhs)")
    }

    public func dotable(shape: Shape) -> Bool {
        col == shape.row
    }

    public func doted(_ shape: Shape) -> Shape {
        Shape(row: row, col: shape.col)
    }
}

public struct Matrix<T:MatrixNumber>: ExpressibleByArrayLiteral & ExpressibleByIntegerLiteral
    & ExpressibleByFloatLiteral & CustomStringConvertible & Equatable {

    enum MatrixError: Error {
        case broadcastingError(_ msg: String)
    }
    public typealias FloatLiteralType = MatrixDefType
    public typealias IntegerLiteralType = Int
    public typealias ArrayLiteralElement = [T]

    public let isSquare: Bool

    let shape: Shape

    let data: [T]

    public init(floatLiteral value: MatrixDefType) {
        data = [T(convertible: value)]
        shape = Shape(1, 1)
        isSquare = true
    }

    public init(integerLiteral value: Int) {
        data = [T(convertible: value)]
        shape = Shape(1, 1)
        isSquare = true
    }

    public init(arrayLiteral elements: Array<T>...) {
        data = elements.flatMap {$0}
        shape = Shape(row: elements.count, col: (elements.count == 0) ? 0 : elements[0].count)
        isSquare = shape.row == shape.col
    }

    public init(internalData: [[T]]) {
        data = internalData.flatMap {$0}
        shape = Shape(row: internalData.count, col: (internalData.count == 0) ? 0 : internalData[0].count)
        isSquare = shape.row == shape.col
    }

    public init(internalData: [T], dim: Shape) {
        data = internalData
        shape = dim
        isSquare = dim.row == dim.col
    }

    public init(_ number: T) {
        data = [number]
        shape = Shape(1, 1)
        isSquare = true
    }

    public subscript(row: Int, col: Int) -> T {
        let shapeCached = shape

        if row >= shapeCached.row || col >= shapeCached.col || row < 0 || col < 0 {
            fatalError("Matrix index overflow: subscripting at (\(row),\(col)) whuile dim=(\(shapeCached.row),\(shapeCached.col))")
        }
        return getInd(row, col)
    }

    public subscript(row: Int) -> ArraySlice<T> {
        let shapeCached = shape

        if row >= shapeCached.row || row < 0 {
            fatalError("Matrix index overflow: subscripting at (\(row),) whuile dim=(\(shapeCached.row),\(shapeCached.col))")
        }
        return data[(row * shape.col)..<((row + 1) * shape.col)]
    }

    public var description: String {
        var res = "("
        for i in 0..<shape.row {
            res += "| "
            for j in 0..<shape.col {
                res += "\(data[i * shape.col + j]) "
            }
            res += "|"
            if i != shape.row - 1 {
                res += "\n"
            }
        }
        res += ", dtype=\(T.self), shape=(\(shape.row), \(shape.col)))"
        return res
    }

    static func setInd(data: inout [T], shape: Shape, _ row: Int, _ col: Int, val: T) {
        data[row * shape.col + col] = val
    }

    func getInd(_ row: Int, _ col: Int) -> T {
        data[row * shape.col + col]
    }

    public func `as`<S: MatrixNumber>(_ type: S.Type) -> Matrix<S> {
        if S.self == T.self {
            return self as! Matrix<S>
        }
        return Matrix<S>(internalData: data.map { elem in
                    S(convertible: elem)
            }, dim: self.shape
        )
    }

    static func generalOp<S>(lhs: Matrix<T>, rhs: Matrix<S>, op: (MatrixDefType, MatrixDefType) -> MatrixDefType) -> Matrix<MatrixDefType> {
        if lhs.shape != rhs.shape {
            if let lhsB = try? lhs.broadcast(shape: rhs.shape) {
                return generalOp(lhs: lhsB, rhs: rhs, op: op)
            }
            if let rhsB = try? rhs.broadcast(shape: lhs.shape) {
                return generalOp(lhs: lhs, rhs: rhsB, op: op)
            }
            fatalError("Matrix operation dims mismatch: dim=(\(lhs.shape.row),\(lhs.shape.col)) and dim=(\(rhs.shape.row),\(rhs.shape.col)) and cannot be broadcasted")
        }
        return Matrix<MatrixDefType>(internalData: zip(lhs.as(MatrixDefType.self).data, rhs.data).map {
                MatrixDefType(convertible: op($0, MatrixDefType(convertible: $1)))
        }, dim: lhs.shape)
    }

    static func generalOp(lhs: Matrix, rhs: Matrix, op: (T, T) -> T) -> Matrix {
        if lhs.shape != rhs.shape {
            if let lhsB = try? lhs.broadcast(shape: rhs.shape) {
                return generalOp(lhs: lhsB, rhs: rhs, op: op)
            }
            if let rhsB = try? rhs.broadcast(shape: lhs.shape) {
                return generalOp(lhs: lhs, rhs: rhsB, op: op)
            }
            fatalError("Matrix addition dims mismatch: dim=(\(lhs.shape.row),\(lhs.shape.col)) and dim=(\(rhs.shape.row),\(rhs.shape.col))")
        }
        return Matrix(internalData: zip(lhs.data, rhs.data).map(op), dim: lhs.shape)
    }

    static func generalOpNum(lhs: Matrix, rhs: FloatConvertible, op: (T, T) -> T) -> Matrix {
        return Matrix(internalData: lhs.data.map {op($0, T(convertible: rhs))}, dim: lhs.shape)
    }

    static func generalOpNum(lhs: FloatConvertible, rhs: Matrix, op: (T, T) -> T) -> Matrix {
        return Matrix(internalData: rhs.data.map {op(T(convertible: lhs), $0)}, dim: rhs.shape)
    }

    static func generalOpNum(lhs: Matrix, rhs: FloatConvertible, op: (T, T) -> MatrixDefType) -> Matrix<MatrixDefType> {
        return Matrix<MatrixDefType>(internalData: lhs.data.map {op($0, T(convertible: rhs))}, dim: lhs.shape)
    }

    static func generalOpNum(lhs: FloatConvertible, rhs: Matrix, op: (T, T) -> MatrixDefType) -> Matrix<MatrixDefType> {
        return Matrix<MatrixDefType>(internalData: rhs.data.map {op(T(convertible: lhs), $0)}, dim: rhs.shape)
    }

    public static func +(lhs: Matrix, rhs: Matrix) -> Matrix {
        return generalOp(lhs: lhs, rhs: rhs, op: +)
    }

    public static func +<S>(lhs: Matrix<T>, rhs: Matrix<S>) -> Matrix<MatrixDefType> {
        return generalOp(lhs: lhs, rhs: rhs, op: +)
    }

    public static func +(lhs: Matrix, rhs: FloatConvertible) -> Matrix {
        return generalOpNum(lhs: lhs, rhs: rhs, op: +)
    }

    public static func +(rhs: FloatConvertible, lhs: Matrix) -> Matrix {
        return generalOpNum(lhs: lhs, rhs: rhs, op: +)
    }
    public static func -<S>(lhs: Matrix<T>, rhs: Matrix<S>) -> Matrix<MatrixDefType> {
        return generalOp(lhs: lhs, rhs: rhs, op: -)
    }

    public static func -(lhs: Matrix, rhs: Matrix) -> Matrix {
        return generalOp(lhs: lhs, rhs: rhs, op: -)
    }

    public static func *(lhs: Matrix, rhs: Matrix) -> Matrix {
        return generalOp(lhs: lhs, rhs: rhs, op: *)
    }

    public static func *<S>(lhs: Matrix<T>, rhs: Matrix<S>) -> Matrix<MatrixDefType> {
        return generalOp(lhs: lhs, rhs: rhs, op: *)
    }

    public static func *(lhs: Matrix, rhs: FloatConvertible) -> Matrix {
        return generalOpNum(lhs: lhs, rhs: rhs, op: *)
    }

    public static func *(rhs: FloatConvertible, lhs: Matrix) -> Matrix {
        return generalOpNum(lhs: lhs, rhs: rhs, op: *)
    }

    public static func -(lhs: Matrix, rhs: FloatConvertible) -> Matrix {
        return generalOpNum(lhs: lhs, rhs: rhs, op: -)
    }

    public static func -(rhs: FloatConvertible, lhs: Matrix) -> Matrix {
        return generalOpNum(lhs: rhs, rhs: lhs, op: -)
    }

    public static func /(lhs: Matrix, rhs: Matrix) -> Matrix<MatrixDefType> {
        return generalOp(lhs: lhs, rhs: rhs, op: /)
    }

    public static func /<S>(lhs: Matrix<T>, rhs: Matrix<S>) -> Matrix<MatrixDefType> {
        return generalOp(lhs: lhs, rhs: rhs, op: /)
    }

    public static func /(lhs: Matrix, rhs: FloatConvertible) -> Matrix<MatrixDefType> {
        return generalOpNum(lhs: lhs, rhs: T(convertible: rhs)) {
             MatrixDefType(convertible: $0) / MatrixDefType(convertible: $1)
        }
    }

    public static func /(lhs: FloatConvertible, rhs: Matrix) -> Matrix<MatrixDefType> {
        return generalOpNum(lhs: T(convertible: lhs), rhs: rhs) {
            MatrixDefType(convertible: $0) / MatrixDefType(convertible: $1)
       }
    }

    public static func **(lhs: Matrix, rhs: FloatConvertible) -> Matrix {
        // TODO: binpow
        return Matrix(internalData: lhs.data.map {
            T(convertible: Foundation.pow(MatrixDefType(convertible: $0), MatrixDefType(convertible: rhs)))
        }, dim: lhs.shape)
    }

    public static prefix func -(lhs: Matrix) -> Matrix {
        return Matrix(internalData: lhs.data.map {
            -1 * $0
        }, dim: lhs.shape)
    }

    public static func zero(dim: Shape) -> Matrix<T> {
        return Matrix<T>.generated(dim: dim) {_, _ in
            0
        }
    }

    public static func zero(dim: Int) -> Matrix<T> {
        return Matrix<T>.zero(dim: Shape(dim, dim))
    }

    public static func ones(dim: Shape) -> Matrix<T> {
        return Matrix<T>.generated(dim: dim) {_, _ in
            1
        }
    }

    public static func ones(dim: Int) -> Matrix<T> {
        return Matrix<T>.ones(dim: Shape(dim, dim))
    }

    public static func eye(dim: Int) -> Matrix<T> {
        return Matrix<T>.generated(dim: Shape(dim, dim)) {i, j in
            (i == j) ? 1: 0
        }
    }

    public static func random(dim: Shape, generator: () -> T) -> Matrix<T> {
        return Matrix<T>.generated(dim: dim) {_, _ in
            generator()
        }
    }

    public static func generated(dim: Shape, generator: (Int, Int) -> T) -> Matrix<T> {
        let data = Array(repeating: T.zero, count: dim.col * dim.row)
        return Matrix<T>(internalData:
                            zip(data, 0..<(dim.col * dim.row)).map {_, ind in generator(ind / dim.col, ind % dim.col)},
                         dim: dim)
    }

    var `T`:Matrix {
        let col = shape.col
        let row = shape.row
        if isSquare {
            var dataNew = data
            for i in 0..<shape.col {
                for j in 0..<i {
                    (dataNew[i * col + j], dataNew[j * col + i]) = (dataNew[j * col + i], dataNew[i * col + j])
                }
            }
            return Matrix(internalData: dataNew, dim: shape)
        } else {
            var dataNew: [T] = Array(repeating: T.zero, count: col * row)
            for i in 0..<row {
                for j in 0..<col {
                    dataNew[i * col + j] = data[j * row + i]
                }
            }
            return Matrix(internalData: dataNew, dim: Shape(row: col, col: row))
        }
    }

    public func dot(_ rhs: Matrix<T>) -> Matrix<T> {
        if !shape.dotable(shape: rhs.shape) {
            fatalError("Matrix dot dims mismatch: dim=(\(shape.row),\(shape.col)) and dim=(\(rhs.shape.row),\(rhs.shape.col)) cause \(shape.col) != \(rhs.shape.row)")
        }
        var dataRes = Array(repeating: T.zero, count: shape.row * rhs.shape.col)

        for i in 0..<shape.row {
            for j in 0..<rhs.shape.col {
                var c = T.zero
                for k in 0..<shape.col {
                    c += data[i * shape.col + k] * rhs.data[k * rhs.shape.col + j]
                }
                dataRes[i * rhs.shape.col + j] = c
            }
        }
        return Matrix<T>(internalData: dataRes, dim: shape.doted(rhs.shape))

        #if false
        if MatrixDefType.self == Float.self {
            var dataRes = Array(repeating: MatrixDefType.zero, count: shape.row * rhs.shape.col)
            vDSP_mmul(self.as(MatrixDefType.self).data, 1, rhs.as(MatrixDefType.self).data, 1, &dataRes, 1, vDSP_Length(shape.row), vDSP_Length(rhs.shape.col), vDSP_Length(shape.col))
            return Matrix<MatrixDefType>(internalData: dataRes, dim: (row: shape.row, col: rhs.shape.col))
        } else {
            var dataRes = Array(repeating: MatrixDefType.zero, count: shape.row * rhs.shape.col) as! [Double]
            vDSP_mmulD(self.as(MatrixDefType.self).data as! [Double], 1, rhs.as(MatrixDefType.self).data as! [Double], 1, &dataRes, 1, vDSP_Length(shape.row), vDSP_Length(rhs.shape.col), vDSP_Length(shape.col))
            return Matrix<Double>(internalData: dataRes, dim: (row: shape.row, col: rhs.shape.col)) as! Matrix<MatrixDefType>
        }
        #endif
    }

    public func dot<S>(_ rhs: Matrix<S>) -> Matrix<MatrixDefType> {
        if shape.col != rhs.shape.row {
            fatalError("Matrix dot dims mismatch: dim=(\(shape.row),\(shape.col)) and dim=(\(rhs.shape.row),\(rhs.shape.col)) - \(shape.col) != \(rhs.shape.row)")
        }
        var dataRes = Array(repeating: MatrixDefType.zero, count: shape.row * rhs.shape.col)

        for i in 0..<shape.row {
            for j in 0..<rhs.shape.col {
                var c = MatrixDefType.zero
                for k in 0..<shape.col {
                    c += MatrixDefType(convertible: data[i * shape.col + k]) * MatrixDefType(convertible: rhs.data[k * shape.col + j])
                }
                dataRes[i * rhs.shape.col + j] = c
            }
        }
        return Matrix<MatrixDefType>(internalData: dataRes, dim: Shape(row: shape.row, col: rhs.shape.col))
    }

    public static func == (lhs: Matrix<T>, rhs: Matrix<T>) -> Bool {
        lhs.shape == rhs.shape && lhs.data == rhs.data
    }

    public func map(_ apply: (T) -> T) -> Matrix {
        Matrix(internalData: data.map(apply), dim: shape)
    }

    public func broadcast(col: Int) throws -> Matrix {
        if shape.col != 1 {
            throw MatrixError.broadcastingError("Cannot broadcast \(shape) to \(Shape(row: shape.row, col: col))")
        }

        let dataRaw = data.flatMap {Array(repeating: $0, count: col)}
        return Matrix(internalData: dataRaw, dim: Shape(shape.row, col))
    }

    public func broadcast(row: Int) throws -> Matrix {
        if shape.row != 1 {
            throw MatrixError.broadcastingError("Cannot broadcast \(shape) to \(Shape(row: row, col: shape.col))")
        }
        let dataRaw = Array(repeating: data, count: row)
        return Matrix(internalData: dataRaw)
    }

    public func broadcast(row: Int, col: Int) throws -> Matrix {
        if shape.row == row {
            return try broadcast(col: col)
        } else if shape.col == col {
            return try broadcast(row: row)
        } else {
            if shape.row != 1 || shape.col != 1 {
                throw MatrixError.broadcastingError("Cannot broadcast \(shape) to \(Shape(row: row, col: col))")
            }
            let dataRaw = Array(repeating: data[0], count: row * col)
            return Matrix(internalData: dataRaw, dim: Shape(row, col))
        }
    }

    public func broadcast(shape: Shape) throws -> Matrix {
        return try broadcast(row: shape.row, col: shape.col)
    }

    public func broadcastable(shape: Shape) -> Bool {
        self.shape.broadcastable(shape: shape)
    }

    public func broadcastable<S>(matrix: Matrix<S>) -> Bool {
        broadcastable(shape: matrix.shape)
    }

    public func downcast(row: Int, col: Int) throws -> Matrix {
        if shape.row == row && shape.col == col {
            return self
        }
        if row == 1 && col == shape.col {
            return Matrix(internalData: Array(data[0..<col]), dim: Shape(1, col))
        } else if col == 1 && row == shape.row {
            var res: [T] = []
            for i in 0..<row {
                res.append(data[i * shape.col])
            }
            return Matrix(internalData: res, dim: Shape(row, 1))
        } else {
            throw MatrixError.broadcastingError("Cannot downcast \(shape) to \(Shape(row: row, col: col))")
        }
    }

    public func reduceMean(shape: Shape) throws -> Matrix {
        if shape == Shape(1, 1) {
            return (sum() / (self.shape.col * self.shape.row)).as(T.self)
        } else if shape == Shape(1, self.shape.col) {
            return reduceSum(axis: 1)
        } else if shape == Shape(self.shape.row, 1) {
            return reduceSum(axis: 0)
        } else if shape == self.shape {
            return self
        }
        throw MatrixError.broadcastingError("Cannot reduceMean \(self.shape) to \(shape)")
    }
}

extension Matrix {
    func max() -> T {
        return data.max()!
    }

    func reduceSum(axis: Int = -1) -> Matrix {
        if axis == 0 {
            var dataRes = Array(repeating: T.zero, count: shape.row)
            for i in 0..<shape.row {
                dataRes[i] = data[(i * shape.col)..<((i + 1) * shape.col)].reduce(0, +)
            }
            return Matrix(internalData: dataRes, dim: Shape(shape.row, 1))
        } else {
            var dataRes = Array(repeating: T.zero, count: shape.col)
            for i in 0..<shape.col {
                dataRes[i] = data[(i * shape.row)..<((i + 1) * shape.row)].reduce(0, +)
            }
            return Matrix(internalData: dataRes, dim: Shape(1, shape.col))
        }
    }

    func sum() -> Matrix {
        return Matrix(data.reduce(T.zero) {$0 + $1})
    }
}

extension Matrix {
    static func cmpBroadCastable(lhs: Matrix, rhs: Matrix, op: (T, T) -> Bool) -> Matrix<Int> {
        if lhs.shape == rhs.shape {
            return Matrix<Int>(internalData: zip(lhs.data, rhs.data).map {
                l, r in op(l, r) ? 1: 0
            }, dim: lhs.shape)
        } else if lhs.broadcastable(shape: rhs.shape) {
            return cmpBroadCastable(lhs: try! lhs.broadcast(shape: rhs.shape), rhs: rhs, op: op)
        } else if rhs.broadcastable(shape: lhs.shape) {
            return cmpBroadCastable(lhs: lhs, rhs: try! rhs.broadcast(shape: lhs.shape), op: op)
        }
        fatalError("Cannot broadcast in cmp \(lhs.shape) to \(rhs.shape)")
    }

    static func<(lhs: Matrix, rhs: FloatConvertible) -> Matrix<Int> {
        cmpBroadCastable(lhs: lhs, rhs: Matrix(T(convertible: rhs)), op: <)
    }

    static func>(lhs: Matrix, rhs: FloatConvertible) -> Matrix<Int> {
        cmpBroadCastable(lhs: lhs, rhs: Matrix(T(convertible: rhs)), op: >)
    }

    static func==(lhs: Matrix, rhs: FloatConvertible) -> Matrix<Int> {
        cmpBroadCastable(lhs: lhs, rhs: Matrix(T(convertible: rhs)), op: ==)
    }

    static func<(rhs: FloatConvertible, lhs: Matrix) -> Matrix<Int> {
        cmpBroadCastable(lhs: lhs, rhs: Matrix(T(convertible: rhs)), op: >)
    }

    static func>(rhs: FloatConvertible, lhs: Matrix) -> Matrix<Int> {
        cmpBroadCastable(lhs: lhs, rhs: Matrix(T(convertible: rhs)), op: <)
    }

    static func==(rhs: FloatConvertible, lhs: Matrix) -> Matrix<Int> {
        cmpBroadCastable(lhs: lhs, rhs: Matrix(T(convertible: rhs)), op: ==)
    }
}
