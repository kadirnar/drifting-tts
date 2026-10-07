import Foundation
import JavaScriptCore

public enum FrontendError: Error, LocalizedError, Sendable {
    case initialization, missingResource, javascript(String), invalidResult, invalidSplitLength
    public var errorDescription: String? {
        switch self {
        case .initialization: return "Cannot create a JavaScriptCore context"
        case .missingResource: return "The bundled Turkish frontend is missing"
        case .javascript(let reason): return "Turkish frontend: \(reason)"
        case .invalidResult: return "The Turkish frontend returned an invalid value"
        case .invalidSplitLength: return "maxChars must be positive"
        }
    }
}

/// Local Turkish normalization shared with the browser implementation.
/// Keep this non-Sendable object confined to its owning actor or serial executor.
public final class TurkishFrontend {
    private let context: JSContext
    private let api: JSValue

    public init() throws {
        guard let url = Bundle.module.url(forResource: "TurkishFrontend", withExtension: "js") else {
            throw FrontendError.missingResource
        }
        guard let context = JSContext() else { throw FrontendError.initialization }
        let source = try String(contentsOf: url, encoding: .utf8)
        context.evaluateScript(source, withSourceURL: url)
        if let exception = context.exception { throw FrontendError.javascript(exception.toString()) }
        guard let api = context.objectForKeyedSubscript("DriftingText"), api.isObject else {
            throw FrontendError.invalidResult
        }
        self.context = context; self.api = api
    }

    private func call(_ method: String, _ arguments: [Any]) throws -> JSValue {
        context.exception = nil
        let value = api.invokeMethod(method, withArguments: arguments)
        if let exception = context.exception { throw FrontendError.javascript(exception.toString()) }
        guard let value, !value.isUndefined, !value.isNull else { throw FrontendError.invalidResult }
        return value
    }

    public func normalize(_ text: String) throws -> String {
        guard let result = try call("normalize", [text]).toString() else { throw FrontendError.invalidResult }
        return result
    }

    /// Takes already-normalized text; call `normalize` first for model input.
    public func splitSentences(_ text: String, maxChars: Int = 180) throws -> [String] {
        guard maxChars > 0 else { throw FrontendError.invalidSplitLength }
        guard let result = try call("splitSentences", [text, maxChars]).toArray() as? [String] else {
            throw FrontendError.invalidResult
        }
        return result
    }

    public func textToIDs(_ text: String, intersperseBlank: Bool = true, normalized: Bool = false) throws -> [Int] {
        guard let result = try call("textToIds", [text, ["intersperseBlank": intersperseBlank,
                                                        "normalized": normalized]]).toArray() as? [Int] else {
            throw FrontendError.invalidResult
        }
        return result
    }

    public func numberToWords(_ digits: String) throws -> String {
        guard let result = try call("numberToWords", [digits]).toString() else { throw FrontendError.invalidResult }
        return result
    }

    public func ordinalToWords(_ digits: String) throws -> String {
        guard let result = try call("ordinalToWords", [digits]).toString() else { throw FrontendError.invalidResult }
        return result
    }

    public var symbols: [String] { api.forProperty("symbols").toArray() as? [String] ?? [] }
    public var padID: Int { Int(api.forProperty("padID").toInt32()) }
    public var blankID: Int { Int(api.forProperty("blankID").toInt32()) }
}
