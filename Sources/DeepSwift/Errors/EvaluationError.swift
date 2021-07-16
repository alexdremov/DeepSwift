//
//  File.swift
//  
//
//  Created by  Alex Dremov on 13.07.2021.
//

import Foundation

public struct EvaluationError: Error{
    enum Causes {
        case variableNoData
        case variableNotDefined(name: String)
        case constantTypeMismatch
    }
    var cause: Causes
    
    init(reason: Causes) {
        cause = reason
    }
}
