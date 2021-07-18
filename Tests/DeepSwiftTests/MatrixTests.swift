//
//  MatrixTests.swift
//  
//
//  Created by  Alex Dremov on 16.07.2021.
//

import Foundation
import XCTest
@testable import DeepSwift

final class MatrixTests: XCTestCase{
    func testAddition(){
        var a:Matrix<Int> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        var b:Matrix<Int> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        XCTAssertEqual((a + b).data, zip(a.data, b.data).map(+))
        
        a = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        b = [
            [1],
            [4],
            [7]
        ]
        
        var c:Matrix<Int> = [
            [1 + 1, 2 + 1, 3 + 1],
            [4 + 4, 5 + 4, 6 + 4],
            [7 + 7, 8 + 7, 9 + 7]
        ]
        
        let resTest = (a + b)
        XCTAssertEqual(resTest.data, c.data)
        
        a = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        b = [
            [3],
        ]
        
        c = [
            [1 + 3, 2 + 3, 3 + 3],
            [4 + 3, 5 + 3, 6 + 3],
            [7 + 3, 8 + 3, 9 + 3]
        ]
        
        XCTAssertEqual((a + b).data, c.data)
        XCTAssertEqual((a + b[0, 0]).data, c.data)
        
        b = [
            [3, 3, 3]
        ]
        
        XCTAssertEqual((a + b).data, c.data)
        XCTAssertEqual((a + 3).data, c.data)
    }
    
    func testMul(){
        var a:Matrix<Int> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        var b:Matrix<Int> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        XCTAssertEqual((a * b).data, zip(a.data, b.data).map(*))
        
        a = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        b = [
            [1],
            [4],
            [7]
        ]
        
        var c:Matrix<Int> = [
            [1 * 1, 2 * 1, 3 * 1],
            [4 * 4, 5 * 4, 6 * 4],
            [7 * 7, 8 * 7, 9 * 7]
        ]
        
        XCTAssertEqual((a * b).data, c.data)
        
        a = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        b = [
            [3],
        ]
        
        c = [
            [1 * 3, 2 * 3, 3 * 3],
            [4 * 3, 5 * 3, 6 * 3],
            [7 * 3, 8 * 3, 9 * 3]
        ]
        
        XCTAssertEqual((a * b).data, c.data)
        XCTAssertEqual((a * b[0, 0]).data, c.data)
        
        b = [
            [3, 3, 3]
        ]
        
        XCTAssertEqual((a * b).data, c.data)
        XCTAssertEqual((a * 3).data, c.data)
    }
    
    func testDiv(){
        var a:Matrix<Int> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        var b:Matrix<Int> = [
            [1, 1, 1],
            [4, 4, 4],
            [7, 7, 7]
        ]
        
        XCTAssertEqual((a / b).data, zip(a.data, b.data).map({MatrixDefType(convertible: $0) / MatrixDefType(convertible: $1)}))
        
        a = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        b = [
            [1],
            [4],
            [7]
        ]
        
        var c:Matrix<MatrixDefType> = [
            [1.0 / 1.0, 2.0 / 1.0, 3.0 / 1.0],
            [4.0 / 4.0, 5.0 / 4.0, 6.0 / 4.0],
            [7.0 / 7.0, 8.0 / 7.0, 9.0 / 7.0]
        ]
        
        XCTAssertEqual((a / b).data, c.data)
        
        a = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        
        b = [
            [3],
        ]
        
        var t: Matrix<MatrixDefType> = [
            [1.0 / 3.0, 2.0 / 3.0, 3.0 / 3.0],
            [4.0 / 3.0, 5.0 / 3.0, 6.0 / 3.0],
            [7.0 / 3.0, 8.0 / 3.0, 9.0 / 3.0]
        ]
        
        XCTAssertEqual((a / b).data, t.as(MatrixDefType.self).data)
        XCTAssertEqual((a / b[0, 0]).data, t.as(MatrixDefType.self).data)
        
        b = [
            [3, 3, 3]
        ]
        
        XCTAssertEqual((a / b).data, t.as(MatrixDefType.self).data)
        XCTAssertEqual((a / 3).data, t.as(MatrixDefType.self).data)
    }
    
    func testBroadcast() {
        let shapeFirst = Shape(7, 13)
        let shapeSecond = Shape(1, 13)
        let shapeThird = Shape(7, 1)
        let shapeFourth = Shape(1, 1)
        
        XCTAssertTrue(shapeSecond.broadcastable(shape: shapeFirst))
        XCTAssertTrue(shapeThird.broadcastable(shape: shapeFirst))
        XCTAssertTrue(shapeFourth.broadcastable(shape: shapeFirst))
        XCTAssertTrue(shapeFirst.broadcastable(shape: shapeFirst))
    }
    
}
