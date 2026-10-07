import DriftingTTSCore
import Foundation
import MLX

/// A single loaded model. MLX work stays serialized; only host PCM crosses actor boundaries.
public actor DriftingSynthesizer {
    public nonisolated let sampleRate: Int
    private let config: CheckpointConfig
    private let acoustic: AcousticModel
    private let vocoder: BigVGAN
    private let frontend: TurkishFrontend
    private let device: Device
    private var running = false

    /// The cache limit is process-wide. Pass nil to preserve an existing MLX allocator policy.
    public init(modelDirectory: URL, device: Device = .gpu, cacheLimitBytes: Int? = 32 * 1024 * 1024) throws {
        #if targetEnvironment(simulator)
        throw SynthesisError.simulatorUnsupported
        #else
        try Task.checkCancellation()
        let config = try CheckpointConfig.load(from: modelDirectory.appendingPathComponent("config.json"))
        if let limit = cacheLimitBytes { Memory.cacheLimit = max(0, limit) }
        self.config = config
        self.sampleRate = config.sampleRate
        self.device = device
        self.frontend = try TurkishFrontend()
        self.acoustic = try Device.withDefaultDevice(device) {
            try Task.checkCancellation()
            let weights = try loadArrays(url: modelDirectory.appendingPathComponent("model.safetensors"))
                .mapValues { $0.asType(.float32) }
            try Task.checkCancellation()
            eval(Array(weights.values))
            return try AcousticModel(config: config.model, weights: weights)
        }
        self.vocoder = try Device.withDefaultDevice(device) {
            try Task.checkCancellation()
            let weights = try loadArrays(url: modelDirectory.appendingPathComponent("vocoder.safetensors"))
                .mapValues { $0.asType(.float32) }
            try Task.checkCancellation()
            eval(Array(weights.values))
            return try BigVGAN(config: config.vocoder, weights: weights)
        }
        try Task.checkCancellation()
        guard acoustic.nMels == config.nMels, acoustic.vocabularySize == config.nVocab,
              acoustic.speakerCount == config.numSpeakers else {
            throw SynthesisError.invalidCheckpoint
        }
        Memory.clearCache()
        #endif
    }

    /// The callback provides backpressure: the next chunk is generated only after it returns.
    /// Cancellation is checked between bounded GPU operations; already submitted Metal work cannot be interrupted.
    public func synthesize(_ text: String, options: SynthesisOptions = .init(),
                           onChunk: @Sendable (AudioChunk) async throws -> Void) async throws -> SynthesisMetrics {
        try options.validate()
        try Task.checkCancellation()
        guard !running else { throw SynthesisError.busy }
        running = true
        defer { running = false; Memory.clearCache() }
        let start = ProcessInfo.processInfo.systemUptime
        let speaker = try config.speakerID(options.voice)
        let temperature = options.temperature ?? config.temperature
        let scale = config.durationScale(for: speaker) * options.lengthScale
        let normalized = try frontend.normalize(text)
        // Shorter sentence segments bound acoustic attention memory on a phone.
        let sentences = try frontend.splitSentences(normalized, maxChars: 120)
        var key = Device.withDefaultDevice(device) { MLXRandom.key(options.seed) }
        var firstAudio: Double?
        var sampleCount = 0
        for (sentenceIndex, sentence) in sentences.enumerated() {
            try Task.checkCancellation()
            if sentenceIndex > 0 && options.pause > 0 {
                let samples = [Float](repeating: 0, count: Int(options.pause * Double(sampleRate)))
                sampleCount += samples.count
                try await onChunk(AudioChunk(samples: samples, sampleRate: sampleRate, sentenceIndex: sentenceIndex,
                    chunkIndex: -1, isSilence: true, firstAudioSeconds: firstAudio ?? 0,
                    elapsedSeconds: ProcessInfo.processInfo.systemUptime - start))
            }
            let ids = try frontend.textToIDs(sentence, normalized: true)
            let (mel, nextKey) = try makeMel(ids: ids, speaker: speaker, scale: scale, temperature: temperature,
                                           cfgScale: options.cfgScale, key: key)
            key = nextKey
            try Task.checkCancellation()
            let frames = mel.dim(1)
            var offset = 0, chunkIndex = 0
            while offset < frames {
                try Task.checkCancellation()
                let count = chunkIndex == 0 ? min(options.firstChunkFrames, options.chunkFrames) : options.chunkFrames
                let end = min(offset + count, frames)
                let left = max(0, offset - vocoder.contextFrames)
                let right = min(frames, end + vocoder.contextFrames)
                let samples = Device.withDefaultDevice(device) {
                    let decoded = vocoder(mel[0..., left..<right, 0...])
                    let from = (offset - left) * vocoder.hopLength
                    let to = (end - left) * vocoder.hopLength
                    return clip(decoded[0, from..<to], min: -1, max: 1).asArray(Float.self)
                }
                guard samples.count == (end - offset) * config.hopLength,
                      samples.allSatisfy(\.isFinite) else { throw SynthesisError.invalidAudio }
                try Task.checkCancellation()
                let elapsed = ProcessInfo.processInfo.systemUptime - start
                if firstAudio == nil { firstAudio = elapsed }
                sampleCount += samples.count
                try await onChunk(AudioChunk(samples: samples, sampleRate: sampleRate, sentenceIndex: sentenceIndex,
                    chunkIndex: chunkIndex, isSilence: false, firstAudioSeconds: firstAudio!, elapsedSeconds: elapsed))
                offset = end
                chunkIndex += 1
            }
        }
        return SynthesisMetrics(audioSeconds: Double(sampleCount) / Double(sampleRate),
            totalSeconds: ProcessInfo.processInfo.systemUptime - start, ttfaSeconds: firstAudio,
            peakMemoryBytes: Memory.peakMemory)
    }

    private func makeMel(ids: [Int], speaker: Int, scale: Float, temperature: Float,
                         cfgScale: Float, key: MLXArray) throws -> (MLXArray, MLXArray) {
        try Device.withDefaultDevice(device) {
        let spk = MLXArray([Int32(speaker)])
        let (tokens, logw) = acoustic.encode(ids: MLXArray(ids.map(Int32.init)).reshaped([1, ids.count]),
                                             speaker: spk, pitchShift: 0)
        let logs = logw.asArray(Float.self)
        var indices: [Int32] = []
        for (index, value) in logs.enumerated() {
            let predicted = ceilf(expf(value) * scale)
            guard predicted.isFinite, predicted >= 0, predicted <= 1024 else { throw SynthesisError.invalidDuration }
            let duration = Int(predicted)
            guard indices.count + duration <= 1024 else { throw SynthesisError.sentenceTooLong(indices.count + duration) }
            indices.append(contentsOf: repeatElement(Int32(index), count: duration))
        }
        guard !indices.isEmpty else { throw SynthesisError.invalidDuration }
        let condition = take(tokens, MLXArray(indices), axis: 1)
        let keys = MLXRandom.split(key: key, into: 3)
        let noise = MLXRandom.normal([1, indices.count, config.nMels], key: keys[1]) * temperature
        let labels = MLXRandom.randInt(low: 0, high: config.model.gen.noiseClasses,
            [1, max(1, config.model.gen.noiseCoords)], key: keys[2])
        let result = acoustic.generate(noise: noise, condition: condition, speaker: spk,
            cfg: MLXArray([cfgScale]), noiseLabels: labels) * config.stats.std + config.stats.mean
        eval(result)
        return (result, keys[0])
        }
    }
}
