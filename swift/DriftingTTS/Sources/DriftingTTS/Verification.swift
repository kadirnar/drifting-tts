import DriftingTTSCore
import Foundation
import MLX
import MLXRandom

public enum VerificationError: Error, LocalizedError {
    case mismatch(String)
    public var errorDescription: String? {
        switch self { case .mismatch(let reason): "Parity check failed: \(reason)" }
    }
}

/// Offline comparisons with deterministic Python MLX references; suitable for CI and a device diagnostic.
public enum Verification {
    private struct Cases: Decodable, Sendable {
        struct Named: Decodable, Sendable { let name: String }
        struct Encode: Decodable, Sendable { let name: String; let pitchShift: Float }
        struct Pipeline: Decodable, Sendable {
            let text, speaker: String
            let seed: UInt64
            let cfgScale, temperature, lengthScale: Float
            let pause: Double
            let chunkFrames, firstChunkFrames, sampleCount, silenceChunks: Int
        }
        let formatVersion: Int
        let encodeCases: [Encode]
        let generateCases, vocoderCases: [Named]
        let contextFrames, chunkStart, chunkEnd: Int
        let pipeline: Pipeline
    }

    private static func readCases(_ directory: URL) throws -> Cases {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let cases = try decoder.decode(Cases.self, from: Data(contentsOf: directory.appendingPathComponent("cases.json")))
        guard cases.formatVersion == 1 else { throw VerificationError.mismatch("unsupported fixture version") }
        return cases
    }

    private static func compare(_ actual: [Float], _ expected: [Float], name: String,
                                absoluteTolerance: Float = 2e-5, relativeTolerance: Float = 2e-5) throws -> Double {
        guard actual.count == expected.count else {
            throw VerificationError.mismatch("\(name) has \(actual.count) elements; expected \(expected.count)")
        }
        var maximum: Float = 0
        for (index, pair) in zip(actual, expected).enumerated() {
            let (value, reference) = pair
            let error = abs(value - reference)
            guard value.isFinite, reference.isFinite, error <= absoluteTolerance + relativeTolerance * abs(reference) else {
                throw VerificationError.mismatch("\(name)[\(index)] is \(value); expected \(reference), error \(error)")
            }
            maximum = max(maximum, error)
        }
        return Double(maximum)
    }

    /// Validate acoustic, vocoder, context cropping, and explicit random-key parity on the chosen device.
    public static func run(directory: URL, device: Device = .cpu) throws -> [String: Double] {
        let config = try CheckpointConfig.load(from: directory.appendingPathComponent("config.json"))
        let cases = try readCases(directory)
        return try Device.withDefaultDevice(device) {
            let model = try AcousticModel(config: config.model,
                weights: loadArrays(url: directory.appendingPathComponent("model.safetensors")))
            let vocoder = try BigVGAN(config: config.vocoder,
                weights: loadArrays(url: directory.appendingPathComponent("vocoder.safetensors")))
            let references = try loadArrays(url: directory.appendingPathComponent("references.safetensors"))
            func tensor(_ name: String) throws -> MLXArray {
                guard let value = references[name] else { throw VerificationError.mismatch("missing reference \(name)") }
                return value
            }
            var errors: [String: Double] = [:]
            func check(_ name: String, _ actual: MLXArray, _ referenceName: String,
                       tolerance: Float = 2e-5) throws {
                let expected = try tensor(referenceName)
                guard actual.shape == expected.shape else {
                    throw VerificationError.mismatch("\(name) has shape \(actual.shape); expected \(expected.shape)")
                }
                errors[name] = try compare(actual.asType(.float32).asArray(Float.self),
                    expected.asType(.float32).asArray(Float.self), name: name, absoluteTolerance: tolerance)
            }
            for item in cases.encodeCases {
                let prefix = "encode.\(item.name)"
                let output = try model.encode(ids: tensor(prefix + ".ids"), speaker: tensor(prefix + ".speaker"),
                                              pitchShift: item.pitchShift)
                try check(prefix + ".tokens", output.tokens, prefix + ".tokens")
                try check(prefix + ".log_durations", output.logDurations, prefix + ".log_durations")
            }
            for item in cases.generateCases {
                let prefix = "generate.\(item.name)"
                let output = try model.generate(noise: tensor(prefix + ".noise"), condition: tensor(prefix + ".condition"),
                    speaker: tensor(prefix + ".speaker"), cfg: tensor(prefix + ".cfg"), noiseLabels: tensor(prefix + ".labels"))
                try check(prefix, output, prefix + ".output")
            }
            for item in cases.vocoderCases {
                let prefix = "vocoder.\(item.name)"
                try check(prefix, vocoder(tensor(prefix + ".mel")), prefix + ".output", tolerance: 2e-6)
            }
            guard vocoder.contextFrames == cases.contextFrames else {
                throw VerificationError.mismatch("context is \(vocoder.contextFrames); expected \(cases.contextFrames)")
            }
            let left = cases.chunkStart - vocoder.contextFrames
            let right = cases.chunkEnd + vocoder.contextFrames
            let mel = try tensor("chunk.mel")
            let decoded = vocoder(mel[0..., left..<right, 0...])
            let start = vocoder.contextFrames * vocoder.hopLength
            let end = (vocoder.contextFrames + cases.chunkEnd - cases.chunkStart) * vocoder.hopLength
            try check("vocoder.context_chunk", decoded[0..., start..<end], "chunk.output", tolerance: 2e-6)
            let keys = MLXRandom.split(key: MLXRandom.key(cases.pipeline.seed), into: 3)
            guard keys[0].asArray(UInt32.self) == (try tensor("random.next_key")).asArray(UInt32.self) else {
                throw VerificationError.mismatch("Threefry key splitting differs from Python")
            }
            errors["random.next_key"] = 0
            try check("random.noise", MLXRandom.normal([1, 17, 100], key: keys[1]), "random.noise", tolerance: 0)
            try check("random.labels", MLXRandom.randInt(low: 0, high: 8, [1, 3], key: keys[2]), "random.labels", tolerance: 0)
            return errors
        }
    }

    private actor ChunkCollector {
        private var chunks: [AudioChunk] = []
        func append(_ chunk: AudioChunk) { chunks.append(chunk) }
        func result() -> [AudioChunk] { chunks }
    }

    /// Exercise public synthesis, sentence pauses, streamed sample accounting, and seeded waveform parity.
    public static func runPipeline(directory: URL, device: Device = .cpu) async throws -> [String: Double] {
        let cases = try readCases(directory)
        let item = cases.pipeline
        let expected = try Device.withDefaultDevice(device) {
            let values = try loadArrays(url: directory.appendingPathComponent("references.safetensors"))
            guard let waveform = values["pipeline.waveform"] else {
                throw VerificationError.mismatch("missing pipeline waveform")
            }
            return waveform.asArray(Float.self)
        }
        let synth = try DriftingSynthesizer(modelDirectory: directory, device: device)
        let collector = ChunkCollector()
        let options = SynthesisOptions(voice: item.speaker, temperature: item.temperature, cfgScale: item.cfgScale,
            lengthScale: item.lengthScale, seed: item.seed, pause: item.pause,
            chunkFrames: item.chunkFrames, firstChunkFrames: item.firstChunkFrames)
        let metrics = try await synth.synthesize(item.text, options: options) { chunk in
            await collector.append(chunk)
        }
        let chunks = await collector.result()
        let samples = chunks.flatMap(\.samples)
        guard samples.count == item.sampleCount, chunks.filter(\.isSilence).count == item.silenceChunks,
              chunks.first?.isSilence == false, chunks.last?.isSilence == false,
              chunks.allSatisfy({ !$0.samples.isEmpty }), metrics.ttfaSeconds != nil else {
            throw VerificationError.mismatch("pipeline sample count, silence placement, or TTFA metadata")
        }
        let error = try compare(samples, expected, name: "pipeline.waveform", absoluteTolerance: 2e-6)
        return ["pipeline.waveform": error]
    }
}
