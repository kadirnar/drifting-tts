import Foundation
import MLX

enum TensorWeightError: Error, LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): message
        }
    }
}

/// Validates the exact Python safetensors contract before any model evaluation.
/// Float16 storage is promoted to Float32; inference does not quantize the model.
struct TensorWeights {
    private var remaining: [String: MLXArray]
    private var checked: [String: MLXArray] = [:]

    init(_ arrays: [String: MLXArray]) {
        remaining = arrays
    }

    @discardableResult
    mutating func require(_ name: String, _ shape: [Int]) throws -> MLXArray {
        guard let value = remaining.removeValue(forKey: name) else {
            throw TensorWeightError.invalid("Missing tensor: \(name)")
        }
        guard value.dtype.isFloatingPoint else {
            throw TensorWeightError.invalid("Tensor \(name) must contain floating-point weights.")
        }
        guard value.ndim == shape.count,
              zip(value.shape, shape).allSatisfy({ actual, expected in
                  actual > 0 && (expected == -1 || actual == expected)
              }) else {
            throw TensorWeightError.invalid("Tensor \(name) has shape \(value.shape), expected \(shape).")
        }
        let result = value.asType(.float32)
        checked[name] = result
        return result
    }

    mutating func linear(_ name: String, input: Int, output: Int) throws {
        try require(name + ".weight", [output, input])
        try require(name + ".bias", [output])
    }

    mutating func convolution(
        _ name: String, input: Int, output: Int, kernel: Int, bias: Bool = true
    ) throws {
        try require(name + ".weight", [output, kernel, input])
        if bias { try require(name + ".bias", [output]) }
    }

    mutating func layerNorm(_ name: String, width: Int) throws {
        try require(name + ".weight", [width])
        try require(name + ".bias", [width])
    }

    func finish() throws -> [String: MLXArray] {
        guard remaining.isEmpty else {
            throw TensorWeightError.invalid("Unexpected tensors: \(remaining.keys.sorted().joined(separator: ", "))")
        }
        eval(Array(checked.values))
        return checked
    }
}

/// Positive divisor; unlike Swift integer division, round toward negative infinity.
func tensorFloorDivide(_ numerator: Int, _ denominator: Int) -> Int {
    let quotient = numerator / denominator
    return numerator % denominator < 0 ? quotient - 1 : quotient
}

func tensorCeilDivide(_ numerator: Int, _ denominator: Int) -> Int {
    -tensorFloorDivide(-numerator, denominator)
}
