import DriftingTTSCore
import Foundation
import MLX
import MLXNN

/// Float32, single-utterance port of drifting_tts/mlx/model.py.
/// Weights retain their original safetensors names and channels-last convolution layout.
final class AcousticModel {
    let config: ModelConfig
    let nMels: Int
    let vocabularySize: Int
    let speakerCount: Int
    private let weights: [String: MLXArray]
    private let cfgFrequencies: MLXArray

    init(config: ModelConfig, weights: [String: MLXArray]) throws {
        self.config = config
        let t = config.text
        let g = config.gen
        guard t.d > 0, t.heads > 0, t.d % t.heads == 0, (t.d / t.heads) % 2 == 0,
              g.hidden > 0, g.heads > 0, g.hidden % g.heads == 0,
              (g.hidden / g.heads) % 2 == 0, g.patch > 0,
              t.layers >= 0, g.depth >= 0, g.nRegisters >= 0, g.noiseCoords >= 0,
              g.noiseClasses > 0, g.numSteps == 1 else {
            throw TensorWeightError.invalid("Unsupported acoustic architecture: expected one-step drifting and even attention head dimensions.")
        }
        var schema = TensorWeights(weights)
        let mel = try schema.require("encoder.proj_mu.weight", [-1, 1, t.d])
        nMels = mel.shape[0]
        let embedding = try schema.require("encoder.emb.weight", [-1, t.d])
        vocabularySize = embedding.shape[0]
        let speakers = try schema.require("encoder.spk.weight", [-1, t.spkDim])
        speakerCount = speakers.shape[0]
        try schema.require("encoder.proj_mu.bias", [nMels])
        try schema.linear("encoder.spk_proj", input: t.spkDim, output: t.d)
        for i in 0..<3 {
            try schema.convolution("encoder.prenet.\(i)", input: t.d, output: t.d, kernel: 5)
            try schema.layerNorm("encoder.prenet_norms.\(i)", width: t.d)
        }
        for i in 0..<t.layers {
            let name = "encoder.layers.\(i)"
            try schema.layerNorm(name + ".norm1", width: t.d)
            try schema.layerNorm(name + ".norm2", width: t.d)
            try schema.linear(name + ".qkv", input: t.d, output: 3 * t.d)
            try schema.linear(name + ".out", input: t.d, output: t.d)
            try schema.convolution(name + ".ffn.conv1", input: t.d, output: t.ffn, kernel: 3)
            try schema.convolution(name + ".ffn.conv2", input: t.ffn, output: t.d, kernel: 3)
        }
        try schema.layerNorm("encoder.norm", width: t.d)
        try Self.validatePredictor("encoder.duration", input: t.d + t.spkDim, schema: &schema)
        if config.pitch.enabled {
            try Self.validatePredictor("pitch_predictor", input: t.d + t.spkDim, schema: &schema)
            try schema.convolution("pitch_emb", input: 1, output: t.d, kernel: 3)
            try schema.require("lf0_stats", [2])
        }
        let hidden = g.hidden
        try schema.linear("generator.in_proj", input: (2 * nMels + t.d) * g.patch, output: hidden)
        try schema.require("generator.spk.weight", [speakerCount, hidden])
        for i in 0..<g.noiseCoords {
            try schema.require("generator.noise_embeds.\(i).weight", [g.noiseClasses, hidden])
        }
        try schema.linear("generator.cfg_embed.mlp.0", input: 256, output: hidden)
        try schema.linear("generator.cfg_embed.mlp.2", input: hidden, output: hidden)
        try schema.require("generator.cfg_norm.weight", [hidden])
        if g.nRegisters > 0 {
            try schema.linear("generator.reg_proj", input: hidden, output: hidden)
            try schema.require("generator.registers", [1, g.nRegisters, hidden])
        }
        let feedForward = (Int((2.0 / 3.0) * Double(hidden) * Double(g.mlpRatio)) + 31) / 32 * 32
        for i in 0..<g.depth {
            let name = "generator.blocks.\(i)"
            try schema.require(name + ".norm1.weight", [hidden])
            try schema.require(name + ".norm2.weight", [hidden])
            try schema.require(name + ".attn.q_norm.weight", [hidden / g.heads])
            try schema.require(name + ".attn.k_norm.weight", [hidden / g.heads])
            try schema.linear(name + ".attn.qkv", input: hidden, output: 3 * hidden)
            try schema.linear(name + ".attn.proj", input: hidden, output: hidden)
            try schema.linear(name + ".mlp.w1", input: hidden, output: feedForward)
            try schema.linear(name + ".mlp.w3", input: hidden, output: feedForward)
            try schema.linear(name + ".mlp.w2", input: feedForward, output: hidden)
            try schema.linear(name + ".ada", input: hidden, output: 6 * hidden)
        }
        try schema.require("generator.final.norm.weight", [hidden])
        try schema.linear("generator.final.ada", input: hidden, output: 2 * hidden)
        try schema.linear("generator.final.linear", input: hidden, output: nMels * g.patch)
        self.weights = try schema.finish()
        cfgFrequencies = exp(-Float(log(10000.0)) * MLXArray(0..<128).asType(.float32) / Float(128))
        eval(cfgFrequencies)
    }

    private static func validatePredictor(_ name: String, input: Int, schema: inout TensorWeights) throws {
        try schema.convolution(name + ".conv1", input: input, output: 256, kernel: 3)
        try schema.convolution(name + ".conv2", input: 256, output: 256, kernel: 3)
        try schema.layerNorm(name + ".norm1", width: 256)
        try schema.layerNorm(name + ".norm2", width: 256)
        try schema.convolution(name + ".proj", input: 256, output: 1, kernel: 1)
    }

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        addMM(weights[name + ".bias"]!, x, weights[name + ".weight"]!.transposed())
    }

    private func convolution(_ x: MLXArray, _ name: String, padding: Int = 0) -> MLXArray {
        conv1d(x, weights[name + ".weight"]!, padding: padding) + weights[name + ".bias"]!
    }

    private func layerNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: weights[name + ".weight"]!, bias: weights[name + ".bias"]!, eps: 1e-5)
    }

    private func rmsNorm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weights[name + ".weight"]!, eps: 1e-6)
    }

    private func predictor(_ x: MLXArray, _ name: String) -> MLXArray {
        let first = layerNorm(relu(convolution(x, name + ".conv1", padding: 1)), name + ".norm1")
        let second = layerNorm(relu(convolution(first, name + ".conv2", padding: 1)), name + ".norm2")
        return convolution(second, name + ".proj")
    }

    /// Deliberately preserve the Python pow-based RoPE calculation; fast RoPE has different rounding.
    private func rotary(_ x: MLXArray) -> MLXArray {
        let count = x.dim(-2)
        let half = x.dim(-1) / 2
        let frequencies = 1.0 / pow(Float(10000), MLXArray(0..<half).asType(.float32) / Float(half))
        let angles = MLXArray(0..<count).asType(.float32)[.ellipsis, .newAxis] * frequencies
        let first = x[.ellipsis, 0..<half]
        let second = x[.ellipsis, half...]
        let cosine = cos(angles)
        let sine = sin(angles)
        return concatenated([first * cosine - second * sine, first * sine + second * cosine], axis: -1)
    }

    private func attention(_ qkv: MLXArray, heads: Int, normPrefix: String? = nil) -> MLXArray {
        let batch = qkv.shape[0]
        let length = qkv.shape[1]
        let width = qkv.shape[2] / 3
        let packed = qkv.reshaped([batch, length, 3, heads, width / heads]).transposed(2, 0, 3, 1, 4)
        var query = packed[0]
        var key = packed[1]
        if let normPrefix {
            query = rmsNorm(query, normPrefix + ".q_norm")
            key = rmsNorm(key, normPrefix + ".k_norm")
        }
        query = rotary(query)
        key = rotary(key)
        let result = MLXFast.scaledDotProductAttention(
            queries: query, keys: key, values: packed[2], scale: 1.0 / sqrt(Float(width / heads)), mask: nil
        )
        return result.transposed(0, 2, 1, 3).reshaped([batch, length, width])
    }

    func encode(ids: MLXArray, speaker: MLXArray, pitchShift: Float) -> (tokens: MLXArray, logDurations: MLXArray) {
        let t = config.text
        let speakerEmbedding = weights["encoder.spk.weight"]!.take(speaker, axis: 0)
        var x = weights["encoder.emb.weight"]!.take(ids, axis: 0) * sqrt(Float(t.d))
            + linear(speakerEmbedding, "encoder.spk_proj").expandedDimensions(axis: 1)
        for i in 0..<3 {
            x = x + layerNorm(relu(convolution(x, "encoder.prenet.\(i)", padding: 2)), "encoder.prenet_norms.\(i)")
        }
        for i in 0..<t.layers {
            let name = "encoder.layers.\(i)"
            x = x + linear(attention(linear(layerNorm(x, name + ".norm1"), name + ".qkv"), heads: t.heads), name + ".out")
            let ff = convolution(gelu(convolution(layerNorm(x, name + ".norm2"), name + ".ffn.conv1", padding: 1)),
                                 name + ".ffn.conv2", padding: 1)
            x = x + ff
        }
        var features = layerNorm(x, "encoder.norm")
        let mean = convolution(features, "encoder.proj_mu")
        let spk = broadcast(speakerEmbedding.expandedDimensions(axis: 1),
                            to: [features.shape[0], features.shape[1], t.spkDim])
        let conditioning = concatenated([features, spk], axis: -1)
        let durations = predictor(conditioning, "encoder.duration")[.ellipsis, 0]
        if config.pitch.enabled {
            var pitch = predictor(conditioning, "pitch_predictor")
            if pitchShift != 0 {
                pitch = pitch + (pitchShift / 12 * Float(log(2.0))) / weights["lf0_stats"]![1]
            }
            features = features + convolution(pitch, "pitch_emb", padding: 1)
        }
        return (concatenated([mean, features], axis: -1), durations)
    }

    private func modulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
        x * (1 + scale.expandedDimensions(axis: 1)) + shift.expandedDimensions(axis: 1)
    }

    func generate(noise: MLXArray, condition: MLXArray, speaker: MLXArray, cfg: MLXArray,
                  noiseLabels: MLXArray) -> MLXArray {
        let g = config.gen
        let batch = noise.shape[0]
        let frames = noise.shape[1]
        var x = concatenated([noise, condition], axis: -1)
        let padding = (g.patch - frames % g.patch) % g.patch
        if padding > 0 { x = padded(x, widths: [[0, 0], [0, padding], [0, 0]]) }
        let patches = x.shape[1] / g.patch
        // PyTorch patch ordering is channel-major, then the position inside each patch.
        x = linear(x.reshaped([batch, patches, g.patch, -1]).transposed(0, 1, 3, 2)
            .reshaped([batch, patches, -1]), "generator.in_proj")
        var embedding = weights["generator.spk.weight"]!.take(speaker, axis: 0)
        for i in 0..<g.noiseCoords {
            embedding = embedding + weights["generator.noise_embeds.\(i).weight"]!.take(noiseLabels[0..., i], axis: 0)
        }
        let angles = cfg.expandedDimensions(axis: 1) * cfgFrequencies
        let sinusoidal = concatenated([cos(angles), sin(angles)], axis: -1)
        let cfgEmbedding = linear(silu(linear(sinusoidal, "generator.cfg_embed.mlp.0")), "generator.cfg_embed.mlp.2")
        embedding = embedding + 0.02 * rmsNorm(cfgEmbedding, "generator.cfg_norm")
        if g.nRegisters > 0 {
            let registers = linear(embedding, "generator.reg_proj").expandedDimensions(axis: 1)
                + weights["generator.registers"]!
            x = concatenated([registers, x], axis: 1)
        }
        let activatedEmbedding = silu(embedding)
        for i in 0..<g.depth {
            let name = "generator.blocks.\(i)"
            let parts = split(linear(activatedEmbedding, name + ".ada"), parts: 6, axis: -1)
            let normalized = modulate(rmsNorm(x, name + ".norm1"), shift: parts[0], scale: parts[1])
            let attended = attention(linear(normalized, name + ".attn.qkv"), heads: g.heads, normPrefix: name + ".attn")
            x = x + parts[2].expandedDimensions(axis: 1) * linear(attended, name + ".attn.proj")
            let normalizedFF = modulate(rmsNorm(x, name + ".norm2"), shift: parts[3], scale: parts[4])
            let ff = linear(silu(linear(normalizedFF, name + ".mlp.w1")) * linear(normalizedFF, name + ".mlp.w3"),
                            name + ".mlp.w2")
            x = x + parts[5].expandedDimensions(axis: 1) * ff
        }
        let final = split(linear(activatedEmbedding, "generator.final.ada"), parts: 2, axis: -1)
        x = linear(modulate(rmsNorm(x, "generator.final.norm"), shift: final[0], scale: final[1]), "generator.final.linear")
        x = x[0..., g.nRegisters..., 0...]
        x = x.reshaped([batch, patches, nMels, g.patch]).transposed(0, 1, 3, 2)
            .reshaped([batch, patches * g.patch, nMels])
        x = x[0..., 0..<frames, 0...]
        return g.residualPrior ? x + condition[.ellipsis, 0..<nMels] : x
    }
}
