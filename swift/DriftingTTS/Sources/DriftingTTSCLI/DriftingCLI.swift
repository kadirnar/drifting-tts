import DriftingTTS
import Foundation
import MLX

@main
struct DriftingCLI {
    static func main() async throws {
        let args = Array(CommandLine.arguments.dropFirst())
        func option(_ key: String) -> String? {
            guard let index = args.firstIndex(of: key), index + 1 < args.count else { return nil }
            return args[index + 1]
        }
        if let path = option("--metallib") { GPU.metallib = URL(fileURLWithPath: path) }
        let device: Device = args.contains("--cpu") ? .cpu : .gpu
        if let path = option("--verify") {
            let directory = URL(fileURLWithPath: path)
            var errors = try Verification.run(directory: directory, device: device)
            errors.merge(try await Verification.runPipeline(directory: directory, device: device)) { _, new in new }
            errors.merge(try await Verification.runLifecycle(directory: directory, device: device)) { _, new in new }
            print(String(data: try JSONEncoder().encode(errors), encoding: .utf8)!)
            return
        }
        guard let model = option("--model") else {
            print("Usage: drifting-tts-swift --model /path/to/mlx [--text TEXT] [--voice studio] [--out speech.wav] "
                  + "[--runs 1] [--chunk-frames 128] [--first-chunk-frames 24] "
                  + "[--metallib /path/to/mlx.metallib] [--cpu]; or --verify /path/to/Fixtures [--cpu]")
            return
        }
        let loadStarted = ProcessInfo.processInfo.systemUptime
        let synth = try DriftingSynthesizer(modelDirectory: URL(fileURLWithPath: model),
                                           device: device)
        print(String(format: "model load: %.3f seconds", ProcessInfo.processInfo.systemUptime - loadStarted))
        var options = SynthesisOptions(voice: option("--voice") ?? "studio")
        if let value = option("--chunk-frames") { options.chunkFrames = Int(value) ?? 0 }
        if let value = option("--first-chunk-frames") { options.firstChunkFrames = Int(value) ?? 0 }
        let runs = option("--runs").flatMap(Int.init) ?? 1
        guard (1...100).contains(runs) else { throw SynthesisError.invalidOptions }
        let output = URL(fileURLWithPath: option("--out") ?? "speech.wav")
        for run in 0..<runs {
            let accumulator = AudioAccumulator(sampleRate: synth.sampleRate)
            let metrics = try await synth.synthesize(option("--text") ?? "Merhaba, nasılsınız?", options: options) { chunk in
                await accumulator.append(chunk.samples)
                if chunk.sentenceIndex == 0 && chunk.chunkIndex == 0 {
                    print(String(format: "run %d first PCM: %.1f ms", run, chunk.firstAudioSeconds * 1000))
                }
            }
            print(String(data: try JSONEncoder().encode(metrics), encoding: .utf8)!)
            if run == runs - 1 { try await accumulator.wav().write(to: output, options: .atomic) }
        }
        print("Wrote \(output.path)")
    }
}

private actor AudioAccumulator {
    let sampleRate: Int
    var samples: [Float] = []
    init(sampleRate: Int) { self.sampleRate = sampleRate }
    func append(_ chunk: [Float]) { samples.append(contentsOf: chunk) }
    func wav() -> Data {
        var data = Data()
        func tag(_ value: String) { data.append(contentsOf: value.utf8) }
        func integer<T: FixedWidthInteger>(_ value: T) {
            var value = value.littleEndian
            withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
        }
        tag("RIFF"); integer(UInt32(36 + 2 * samples.count)); tag("WAVEfmt ")
        integer(UInt32(16)); integer(UInt16(1)); integer(UInt16(1))
        integer(UInt32(sampleRate)); integer(UInt32(sampleRate * 2)); integer(UInt16(2)); integer(UInt16(16))
        tag("data"); integer(UInt32(2 * samples.count))
        for sample in samples { integer(Int16(max(-1, min(1, sample)) * 32767)) }
        return data
    }
}
