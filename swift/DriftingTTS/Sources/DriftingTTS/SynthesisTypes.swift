import Foundation

public struct SynthesisOptions: Sendable {
    public var voice: String
    public var temperature: Float?
    public var cfgScale: Float
    public var lengthScale: Float
    public var seed: UInt64
    public var pause: Double
    public var chunkFrames: Int
    public var firstChunkFrames: Int

    public init(voice: String = "studio", temperature: Float? = nil, cfgScale: Float = 2,
                lengthScale: Float = 1, seed: UInt64 = 0, pause: Double = 0.15,
                chunkFrames: Int = 128, firstChunkFrames: Int = 24) {
        self.voice = voice
        self.temperature = temperature
        self.cfgScale = cfgScale
        self.lengthScale = lengthScale
        self.seed = seed
        self.pause = pause
        self.chunkFrames = chunkFrames
        self.firstChunkFrames = firstChunkFrames
    }

    func validate() throws {
        guard cfgScale.isFinite, lengthScale.isFinite, lengthScale > 0, lengthScale <= 4,
              pause.isFinite, pause >= 0, pause <= 10,
              temperature.map({ $0.isFinite && $0 >= 0 && $0 <= 5 }) ?? true,
              (1...512).contains(chunkFrames), (1...512).contains(firstChunkFrames) else {
            throw SynthesisError.invalidOptions
        }
    }
}

public struct AudioChunk: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let sentenceIndex: Int
    public let chunkIndex: Int
    public let isSilence: Bool
    public let firstAudioSeconds: Double
    public let elapsedSeconds: Double
}

public struct SynthesisMetrics: Sendable, Codable {
    public let audioSeconds: Double
    public let totalSeconds: Double
    public let ttfaSeconds: Double?
    public let peakMemoryBytes: Int
}

public enum SynthesisError: Error, LocalizedError {
    case invalidOptions
    case invalidCheckpoint
    case unknownVoice(String)
    case busy
    case invalidDuration
    case sentenceTooLong(Int)
    case invalidAudio
    case simulatorUnsupported

    public var errorDescription: String? {
        switch self {
        case .invalidOptions: "Invalid synthesis options. Use positive chunk sizes and a length scale between 0 and 4."
        case .invalidCheckpoint: "Checkpoint tensor dimensions do not match config.json."
        case .unknownVoice(let voice): "Unknown voice: \(voice)."
        case .busy: "A speech request is already running. Cancel it before starting another."
        case .invalidDuration: "The model predicted an invalid speech duration."
        case .sentenceTooLong(let frames): "The sentence needs \(frames) mel frames. Split it into shorter sentences."
        case .invalidAudio: "The model returned invalid audio."
        case .simulatorUnsupported: "MLX inference requires a physical Apple device; use an iPhone or a supported Mac."
        }
    }
}
