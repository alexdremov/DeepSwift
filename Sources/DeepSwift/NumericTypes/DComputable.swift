//
// Created by Alex Dremov on 13.07.2021.
//

import Foundation

public protocol FloatConvertible {
    var doubleValue: Double { get }
    var floatValue: Float { get }
    var intValue: Int { get }
    var CGFloatValue: CGFloat { get }
    init(convertible: FloatConvertible)
}

public extension FloatConvertible {
    var floatValue: Float {get {return Float(doubleValue)}}
    var intValue: Int {get {return Int(doubleValue.rounded())}}
    var CGFloatValue: CGFloat {get {return CGFloat(doubleValue)}}
}

extension FloatConvertible {
    public var description: String {
        String(doubleValue)
    }
}

extension CGFloat: FloatConvertible {
    public var doubleValue : Double {Double(self)}

    public init(convertible: FloatConvertible) {
        self = convertible.CGFloatValue
    }
}

extension Float: FloatConvertible {
    public var doubleValue : Double {Double(self)}

    public init(convertible: FloatConvertible) {
        self = convertible.floatValue
    }
}

extension Double: FloatConvertible {
    public var doubleValue : Double {self}

    public init(convertible: FloatConvertible) {
        self = convertible.doubleValue
    }
}

extension Int: FloatConvertible {
    public var doubleValue : Double {Double(self)}

    public init(convertible: FloatConvertible) {
        self = convertible.intValue
    }
}

public typealias DComputable = FloatConvertible

