import Foundation

public enum ConfigurationError: Error, LocalizedError, Equatable, Sendable {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

public struct TextConfig: Codable, Sendable {
    public let d, heads, layers, ffn, spkDim: Int
    enum CodingKeys: String, CodingKey { case d, heads, layers, ffn; case spkDim = "spk_dim" }
    public init(d: Int, heads: Int, layers: Int, ffn: Int, spkDim: Int) {
        self.d = d; self.heads = heads; self.layers = layers; self.ffn = ffn; self.spkDim = spkDim
    }
}

public struct GeneratorConfig: Codable, Sendable {
    public let hidden, depth, heads, patch, nRegisters, noiseClasses, noiseCoords, numSteps: Int
    public let mlpRatio: Float
    public let residualPrior: Bool
    enum CodingKeys: String, CodingKey {
        case hidden, depth, heads, patch
        case nRegisters = "n_registers", noiseClasses = "noise_classes", noiseCoords = "noise_coords"
        case numSteps = "num_steps", mlpRatio = "mlp_ratio", residualPrior = "residual_prior"
    }
    public init(hidden: Int, depth: Int, heads: Int, patch: Int, mlpRatio: Float, nRegisters: Int,
                noiseClasses: Int, noiseCoords: Int, residualPrior: Bool = true, numSteps: Int = 1) {
        self.hidden = hidden; self.depth = depth; self.heads = heads; self.patch = patch
        self.mlpRatio = mlpRatio; self.nRegisters = nRegisters; self.noiseClasses = noiseClasses
        self.noiseCoords = noiseCoords; self.residualPrior = residualPrior; self.numSteps = numSteps
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hidden = try c.decode(Int.self, forKey: .hidden); depth = try c.decode(Int.self, forKey: .depth)
        heads = try c.decode(Int.self, forKey: .heads); patch = try c.decode(Int.self, forKey: .patch)
        mlpRatio = try c.decode(Float.self, forKey: .mlpRatio)
        nRegisters = try c.decode(Int.self, forKey: .nRegisters)
        noiseClasses = try c.decode(Int.self, forKey: .noiseClasses)
        noiseCoords = try c.decode(Int.self, forKey: .noiseCoords)
        residualPrior = try c.decodeIfPresent(Bool.self, forKey: .residualPrior) ?? true
        numSteps = try c.decodeIfPresent(Int.self, forKey: .numSteps) ?? 1
    }
}

public struct PitchConfig: Codable, Sendable {
    public let enabled: Bool
    public init(enabled: Bool = false) { self.enabled = enabled }
}

public struct ModelConfig: Codable, Sendable {
    public let text: TextConfig
    public let gen: GeneratorConfig
    public let pitch: PitchConfig
    enum CodingKeys: String, CodingKey { case text, gen, pitch }
    public init(text: TextConfig, gen: GeneratorConfig, pitch: PitchConfig = .init()) {
        self.text = text; self.gen = gen; self.pitch = pitch
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(TextConfig.self, forKey: .text)
        gen = try c.decode(GeneratorConfig.self, forKey: .gen)
        pitch = try c.decodeIfPresent(PitchConfig.self, forKey: .pitch) ?? .init()
    }
}

public struct VocoderConfig: Codable, Sendable {
    public let numMels, upsampleInitialChannel: Int
    public let upsampleRates, upsampleKernelSizes, resblockKernelSizes: [Int]
    public let resblockDilationSizes: [[Int]]
    public let resblock, activation: String
    public let snakeLogscale, useTanhAtFinal, useBiasAtFinal: Bool
    enum CodingKeys: String, CodingKey {
        case resblock, activation
        case numMels = "num_mels", upsampleInitialChannel = "upsample_initial_channel"
        case upsampleRates = "upsample_rates", upsampleKernelSizes = "upsample_kernel_sizes"
        case resblockKernelSizes = "resblock_kernel_sizes", resblockDilationSizes = "resblock_dilation_sizes"
        case snakeLogscale = "snake_logscale", useTanhAtFinal = "use_tanh_at_final"
        case useBiasAtFinal = "use_bias_at_final"
    }
    public init(numMels: Int, upsampleRates: [Int], upsampleKernelSizes: [Int], upsampleInitialChannel: Int,
                resblock: String = "1", resblockKernelSizes: [Int], resblockDilationSizes: [[Int]],
                activation: String = "snakebeta", snakeLogscale: Bool = true,
                useTanhAtFinal: Bool = true, useBiasAtFinal: Bool = true) {
        self.numMels = numMels; self.upsampleRates = upsampleRates; self.upsampleKernelSizes = upsampleKernelSizes
        self.upsampleInitialChannel = upsampleInitialChannel; self.resblock = resblock
        self.resblockKernelSizes = resblockKernelSizes; self.resblockDilationSizes = resblockDilationSizes
        self.activation = activation; self.snakeLogscale = snakeLogscale
        self.useTanhAtFinal = useTanhAtFinal; self.useBiasAtFinal = useBiasAtFinal
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numMels = try c.decode(Int.self, forKey: .numMels)
        upsampleRates = try c.decode([Int].self, forKey: .upsampleRates)
        upsampleKernelSizes = try c.decode([Int].self, forKey: .upsampleKernelSizes)
        upsampleInitialChannel = try c.decode(Int.self, forKey: .upsampleInitialChannel)
        resblockKernelSizes = try c.decode([Int].self, forKey: .resblockKernelSizes)
        resblockDilationSizes = try c.decode([[Int]].self, forKey: .resblockDilationSizes)
        resblock = try c.decodeIfPresent(String.self, forKey: .resblock) ?? "1"
        activation = try c.decode(String.self, forKey: .activation)
        snakeLogscale = try c.decode(Bool.self, forKey: .snakeLogscale)
        useTanhAtFinal = try c.decodeIfPresent(Bool.self, forKey: .useTanhAtFinal) ?? true
        useBiasAtFinal = try c.decodeIfPresent(Bool.self, forKey: .useBiasAtFinal) ?? true
    }
}

public struct MelStatistics: Codable, Sendable {
    public let mean, std: Float
    public init(mean: Float, std: Float) { self.mean = mean; self.std = std }
}

/// Metadata exported by `python -m drifting_tts.mlx.convert`.
public struct CheckpointConfig: Codable, Sendable {
    public let model: ModelConfig
    public let vocoder: VocoderConfig
    public let numSpeakers, nVocab, nMels, sampleRate, hopLength: Int
    public let stats: MelStatistics
    public let durationScale, temperature: Float
    public let durationScales: [String: Float]
    public let voices: [String: Int]
    public let defaultVoice: String
    enum CodingKeys: String, CodingKey {
        case model, vocoder, stats, temperature, voices
        case numSpeakers = "num_speakers", nVocab = "n_vocab", nMels = "n_mels"
        case sampleRate = "sample_rate", hopLength = "hop_length", durationScale = "duration_scale"
        case durationScales = "duration_scales", defaultVoice = "default_voice"
    }
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decode(ModelConfig.self, forKey: .model)
        vocoder = try c.decode(VocoderConfig.self, forKey: .vocoder)
        numSpeakers = try c.decode(Int.self, forKey: .numSpeakers)
        nVocab = try c.decode(Int.self, forKey: .nVocab); nMels = try c.decode(Int.self, forKey: .nMels)
        sampleRate = try c.decode(Int.self, forKey: .sampleRate)
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 256
        stats = try c.decode(MelStatistics.self, forKey: .stats)
        durationScale = try c.decodeIfPresent(Float.self, forKey: .durationScale) ?? 1
        durationScales = try c.decodeIfPresent([String: Float].self, forKey: .durationScales) ?? [:]
        temperature = try c.decodeIfPresent(Float.self, forKey: .temperature) ?? 0.3
        voices = try c.decode([String: Int].self, forKey: .voices)
        defaultVoice = try c.decode(String.self, forKey: .defaultVoice)
        try validate()
    }
    public static func load(from url: URL) throws -> CheckpointConfig {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
    public func speakerID(_ voice: String? = nil) throws -> Int {
        let name = voice ?? defaultVoice
        guard let id = voices[name] ?? Int(name), (0..<numSpeakers).contains(id) else {
            throw ConfigurationError.invalid("Unknown speaker: \(name)")
        }
        return id
    }
    public func durationScale(for speaker: Int) -> Float { durationScales[String(speaker)] ?? durationScale }

    public func validate() throws {
        func require(_ value: Bool, _ message: String) throws {
            if !value { throw ConfigurationError.invalid(message) }
        }
        let t = model.text, g = model.gen, v = vocoder
        try require(g.numSteps == 1, "Only one-step checkpoints are supported")
        try require(numSpeakers > 0 && nVocab == 39 && nMels > 0 && sampleRate > 0 && hopLength > 0,
                    "Invalid checkpoint dimensions or Turkish vocabulary")
        try require(t.d > 0 && t.heads > 0 && t.layers > 0 && t.ffn > 0 && t.spkDim > 0,
                    "Text dimensions must be positive")
        try require(t.d % t.heads == 0 && (t.d / t.heads) % 2 == 0, "Text heads need an even rotary dimension")
        try require(g.hidden > 0 && g.heads > 0 && g.depth > 0 && g.patch > 0 && g.mlpRatio.isFinite && g.mlpRatio > 0,
                    "Generator dimensions must be positive")
        try require(g.hidden % g.heads == 0 && (g.hidden / g.heads) % 2 == 0,
                    "Generator heads need an even rotary dimension")
        try require(g.nRegisters >= 0 && g.noiseClasses > 0 && g.noiseCoords >= 0, "Invalid style dimensions")
        try require(v.numMels == nMels && v.upsampleInitialChannel > 0 && v.resblock == "1",
                    "Unsupported vocoder dimensions or residual block")
        try require(["snake", "snakebeta"].contains(v.activation), "Unsupported vocoder activation")
        try require(!v.upsampleRates.isEmpty && v.upsampleRates.count == v.upsampleKernelSizes.count,
                    "Vocoder upsampling stages do not match")
        var hop = 1, channels = v.upsampleInitialChannel
        for (rate, kernel) in zip(v.upsampleRates, v.upsampleKernelSizes) {
            try require(rate > 0 && kernel >= rate && (kernel - rate) % 2 == 0,
                        "Vocoder stages must preserve an exact integer hop")
            let (product, overflow) = hop.multipliedReportingOverflow(by: rate)
            try require(!overflow, "Vocoder hop overflows an integer")
            hop = product; channels /= 2
            try require(channels > 0, "Vocoder has too many upsampling stages for its channel count")
        }
        try require(hop == hopLength, "Vocoder hop does not match checkpoint hop_length")
        try require(!v.resblockKernelSizes.isEmpty && v.resblockKernelSizes.count == v.resblockDilationSizes.count,
                    "Vocoder residual branches do not match")
        for (kernel, dilations) in zip(v.resblockKernelSizes, v.resblockDilationSizes) {
            try require(kernel > 0 && kernel % 2 == 1 && !dilations.isEmpty && dilations.allSatisfy { $0 > 0 },
                        "Invalid vocoder residual kernel or dilation")
        }
        try require(stats.mean.isFinite && stats.std.isFinite && stats.std > 0, "Invalid mel normalization")
        try require(durationScale.isFinite && durationScale > 0 && temperature.isFinite && temperature >= 0,
                    "Invalid duration scale or noise temperature")
        try require(voices[defaultVoice] != nil && voices.values.allSatisfy { (0..<numSpeakers).contains($0) },
                    "Invalid default voice or speaker table")
        for (key, value) in durationScales {
            try require(Int(key).map { (0..<numSpeakers).contains($0) } == true && value.isFinite && value > 0,
                        "Invalid per-speaker duration scale")
        }
    }
}
