import DriftingTTSCore
import Foundation
import MLX

// Float32 Kaiser-sinc coefficients from the Python implementation (12 taps, cutoff .25, half-width .3).
// Keeping this fixed table avoids platform differences in the Bessel/NumPy window implementation.
private let antiAliasFilter: [Float] = [
    0.00202896655537, 0.00938946381211, -0.0255434643477, -0.0576573759317,
    0.128572613001, 0.443209797144, 0.443209797144, 0.128572613001,
    -0.0576573759317, -0.0255434643477, 0.00938946381211, 0.00202896655537,
]

// Only the small elementwise filter/Snake kernels are compiled. Whole-vocoder compilation would
// specialize on lengths and impose unnecessary first-request latency on a phone.
private let upsampleSnake = compile(shapeless: true) { (arrays: [MLXArray]) -> [MLXArray] in
    let alpha = arrays[0]
    let inverseBeta = arrays[1]
    var even = arrays[2] * (2 * antiAliasFilter[11])
    var odd = arrays[3] * (2 * antiAliasFilter[10])
    for i in 1..<6 {
        even = even + arrays[2 + i] * (2 * antiAliasFilter[11 - 2 * i])
        odd = odd + arrays[3 + i] * (2 * antiAliasFilter[10 - 2 * i])
    }
    let evenSine = sin(even * alpha)
    let oddSine = sin(odd * alpha)
    return [even + inverseBeta * (evenSine * evenSine), odd + inverseBeta * (oddSine * oddSine)]
}

private let downsample = compile(shapeless: true) { (arrays: [MLXArray]) -> [MLXArray] in
    var result = arrays[0] * antiAliasFilter[0]
    for i in 1..<6 { result = result + arrays[i] * antiAliasFilter[2 * i] }
    for i in 0..<6 { result = result + arrays[6 + i] * antiAliasFilter[2 * i + 1] }
    return [result]
}

private struct VocoderConvolution {
    let weight: MLXArray
    let bias: MLXArray?
    let padding: Int
    let dilation: Int

    init(_ name: String, input: Int, output: Int, kernel: Int, dilation: Int = 1,
         bias: Bool = true, schema: inout TensorWeights) throws {
        weight = try schema.require(name + ".weight", [output, kernel, input])
        self.bias = bias ? try schema.require(name + ".bias", [output]) : nil
        self.padding = (kernel - 1) * dilation / 2
        self.dilation = dilation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1d(x, weight, padding: padding, dilation: dilation)
        return bias.map { y + $0 } ?? y
    }
}

private struct VocoderActivation {
    let alpha: MLXArray
    let inverseBeta: MLXArray

    init(_ name: String, channels: Int, logarithmic: Bool, hasBeta: Bool, schema: inout TensorWeights) throws {
        let a = try schema.require(name + ".act.alpha", [channels])
        let b = hasBeta ? try schema.require(name + ".act.beta", [channels]) : a
        alpha = logarithmic ? exp(a) : a
        inverseBeta = 1.0 / ((logarithmic ? exp(b) : b) + 1e-9)
        eval(alpha, inverseBeta)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let frames = input.shape[1]
        let x = padded(input, widths: [[0, 0], [3, 3], [0, 0]], mode: .edge)
        var arguments = [alpha, inverseBeta]
        arguments.append(contentsOf: (0..<7).map { x[0..., $0..<($0 + frames), 0...] })
        let phases = upsampleSnake(arguments)
        let first = phases[0][0..., 0..<1, 0...]
        let last = phases[1][0..., (frames - 1)..<frames, 0...]
        func edge(_ value: MLXArray, _ count: Int) -> MLXArray {
            broadcast(value, to: [value.shape[0], count, value.shape[2]])
        }
        let even = concatenated([edge(first, 3), phases[1], edge(last, 2)], axis: 1)
        let odd = concatenated([edge(first, 2), phases[0], edge(last, 3)], axis: 1)
        let shifted = (0..<6).map { even[0..., $0..<($0 + frames), 0...] }
            + (0..<6).map { odd[0..., $0..<($0 + frames), 0...] }
        return downsample(shifted)[0]
    }
}

/// Equivalent stride-one convolution with phases packed into channels; no inserted zeros.
private struct VocoderUpsample {
    let weight: MLXArray
    let bias: MLXArray
    let stride: Int
    let kernel: Int
    let padding: Int
    let taps: Int
    let shift: Int
    let outputChannels: Int

    init(_ name: String, input: Int, output: Int, kernel: Int, stride: Int, schema: inout TensorWeights) throws {
        let original = try schema.require(name + ".weight", [output, kernel, input])
        bias = try schema.require(name + ".bias", [output])
        self.stride = stride
        self.kernel = kernel
        let convolutionPadding = (kernel - stride) / 2
        padding = convolutionPadding
        outputChannels = output
        let lo = (0..<stride).map { -tensorFloorDivide($0 + convolutionPadding, stride) }.min()!
        let hi = (0..<stride).map { tensorFloorDivide(kernel - 1 - $0 - convolutionPadding, stride) }.max()!
        taps = hi - lo + 1
        shift = hi
        let offset = -stride * lo - padding
        let paddedWeight = padded(original, widths: [[0, 0], [offset, stride * taps - kernel - offset], [0, 0]])
        weight = flipped(paddedWeight.reshaped([output, taps, stride, input]), axis: 1)
            .transposed(2, 0, 1, 3).reshaped([stride * output, taps, input])
        eval(weight, bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let frames = x.shape[1]
        let length = (frames - 1) * stride - 2 * padding + kernel
        let positions = tensorCeilDivide(length, stride)
        let right = positions + taps - 1 - frames - shift
        let y: MLXArray
        if shift == right {
            y = conv1d(x, weight, padding: shift)
        } else {
            let paddedInput = padded(x, widths: [[0, 0], [shift, max(right, 0)], [0, 0]])
            y = conv1d(paddedInput[0..., 0..<(positions + taps - 1), 0...], weight)
        }
        return y.reshaped([batch, positions * stride, outputChannels])[0..., 0..<length, 0...] + bias
    }
}

private struct VocoderResidualBlock {
    let first: [VocoderConvolution]
    let second: [VocoderConvolution]
    let activations: [VocoderActivation]

    init(_ name: String, channels: Int, kernel: Int, dilations: [Int],
         logarithmic: Bool, hasBeta: Bool, schema: inout TensorWeights) throws {
        var first = [VocoderConvolution]()
        var second = [VocoderConvolution]()
        var activations = [VocoderActivation]()
        for (i, dilation) in dilations.enumerated() {
            first.append(try VocoderConvolution(name + ".convs1.\(i)", input: channels, output: channels,
                                                kernel: kernel, dilation: dilation, schema: &schema))
            second.append(try VocoderConvolution(name + ".convs2.\(i)", input: channels, output: channels,
                                                 kernel: kernel, schema: &schema))
            for j in 0..<2 {
                activations.append(try VocoderActivation(name + ".activations.\(2 * i + j)",
                    channels: channels, logarithmic: logarithmic, hasBeta: hasBeta, schema: &schema))
            }
        }
        self.first = first
        self.second = second
        self.activations = activations
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for i in first.indices {
            x = x + second[i](activations[2 * i + 1](first[i](activations[2 * i](x))))
        }
        return x
    }
}

/// Float32 BigVGAN-v2 using the same folded weights and polyphase math as the Python backend.
/// Input: [batch, melFrames, melChannels]. Output: [batch, melFrames * hopLength].
final class BigVGAN {
    let hopLength: Int
    let contextFrames: Int
    let lowMemory: Bool
    private let pre: VocoderConvolution
    private let upsample: [VocoderUpsample]
    private let blocks: [[VocoderResidualBlock]]
    private let activation: VocoderActivation
    private let post: VocoderConvolution
    private let useTanh: Bool

    init(config: VocoderConfig, weights: [String: MLXArray], lowMemory: Bool = true) throws {
        guard config.resblock == "1", ["snake", "snakebeta"].contains(config.activation),
              !config.upsampleRates.isEmpty,
              config.upsampleRates.count == config.upsampleKernelSizes.count,
              !config.resblockKernelSizes.isEmpty,
              config.resblockKernelSizes.count == config.resblockDilationSizes.count else {
            throw TensorWeightError.invalid("Unsupported BigVGAN architecture.")
        }
        for (rate, kernel) in zip(config.upsampleRates, config.upsampleKernelSizes) {
            guard rate > 0, kernel >= rate, (kernel - rate) % 2 == 0 else {
                throw TensorWeightError.invalid("BigVGAN upsampling must preserve an exact integer hop.")
            }
        }
        for (kernel, dilations) in zip(config.resblockKernelSizes, config.resblockDilationSizes) {
            guard kernel > 0, kernel % 2 == 1, !dilations.isEmpty, dilations.allSatisfy({ $0 > 0 }) else {
                throw TensorWeightError.invalid("Invalid BigVGAN residual kernels or dilations.")
            }
        }
        hopLength = config.upsampleRates.reduce(1, *)
        contextFrames = Self.requiredContext(config)
        self.lowMemory = lowMemory
        useTanh = config.useTanhAtFinal
        var schema = TensorWeights(weights)
        var channels = config.upsampleInitialChannel
        let hasBeta = config.activation == "snakebeta"
        pre = try VocoderConvolution("conv_pre", input: config.numMels, output: channels, kernel: 7, schema: &schema)
        var upsample = [VocoderUpsample]()
        var blocks = [[VocoderResidualBlock]]()
        for (i, rate) in config.upsampleRates.enumerated() {
            let output = channels / 2
            guard output > 0 else { throw TensorWeightError.invalid("BigVGAN channels exhausted before final stage.") }
            upsample.append(try VocoderUpsample("ups.\(i).0", input: channels, output: output,
                                               kernel: config.upsampleKernelSizes[i], stride: rate, schema: &schema))
            channels = output
            var stage = [VocoderResidualBlock]()
            for j in config.resblockKernelSizes.indices {
                let index = i * config.resblockKernelSizes.count + j
                stage.append(try VocoderResidualBlock("resblocks.\(index)", channels: channels,
                    kernel: config.resblockKernelSizes[j], dilations: config.resblockDilationSizes[j],
                    logarithmic: config.snakeLogscale, hasBeta: hasBeta, schema: &schema))
            }
            blocks.append(stage)
        }
        self.upsample = upsample
        self.blocks = blocks
        activation = try VocoderActivation("activation_post", channels: channels,
            logarithmic: config.snakeLogscale, hasBeta: hasBeta, schema: &schema)
        post = try VocoderConvolution("conv_post", input: channels, output: 1, kernel: 7,
                                      bias: config.useBiasAtFinal, schema: &schema)
        _ = try schema.finish()
    }

    private static func requiredContext(_ config: VocoderConfig) -> Int {
        let radius = zip(config.resblockKernelSizes, config.resblockDilationSizes).map { kernel, dilations in
            dilations.reduce(0) { $0 + 10 + (kernel - 1) * ($1 + 1) / 2 }
        }.max()!
        var lo = -8
        var hi = config.upsampleRates.reduce(1, *) - 1 + 8
        for (stride, kernel) in zip(config.upsampleRates, config.upsampleKernelSizes).reversed() {
            lo -= radius
            hi += radius
            let padding = (kernel - stride) / 2
            lo = tensorCeilDivide(lo + padding - kernel + 1, stride)
            hi = tensorFloorDivide(hi + padding, stride)
        }
        return max(3 - lo, hi + 3)
    }

    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = pre(mel.asType(.float32))
        for i in upsample.indices {
            x = upsample[i](x)
            var sum = blocks[i][0](x)
            if lowMemory { eval(sum) }
            for j in 1..<blocks[i].count {
                sum = sum + blocks[i][j](x)
                if lowMemory { eval(sum) }
            }
            x = sum / Float(blocks[i].count)
            // Materialization bounds live intermediate graphs for the iPhone. It does not change precision.
            if lowMemory { eval(x) }
        }
        let audio = post(activation(x))[.ellipsis, 0]
        return useTanh ? tanh(audio) : audio
    }
}
