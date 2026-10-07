import AVFoundation
import DriftingTTS
import Foundation

/// Main-thread audio scheduling; MLX inference stays in DriftingSynthesizer's actor.
@MainActor
final class PCMPlayer {
    private var engine: AVAudioEngine?
    private var node: AVAudioPlayerNode?
    private var format: AVAudioFormat?
    private var generation = UUID()
    private var scheduledFrames: Int64 = 0
    private var pendingBuffers = 0

    private var playedFrames: Int64 {
        if let node, let time = node.lastRenderTime, let position = node.playerTime(forNodeTime: time) {
            return position.sampleTime
        }
        return 0
    }

    private var queuedSeconds: Double {
        guard let format else { return 0 }
        return Double(max(0, scheduledFrames - playedFrames)) / format.sampleRate
    }

    func enqueue(_ chunk: AudioChunk) async throws {
        try Task.checkCancellation()
        guard !chunk.samples.isEmpty else { return }
        if engine == nil { try start(sampleRate: chunk.sampleRate) }
        let current = generation
        let duration = Double(chunk.samples.count) / Double(chunk.sampleRate)
        // Backpressure limits queued PCM to two seconds; an individual larger chunk is allowed on its own.
        while queuedSeconds + duration > max(2, duration) {
            try await Task.sleep(for: .milliseconds(10))
            guard current == generation else { throw CancellationError() }
        }
        try Task.checkCancellation()
        guard current == generation, let node, let format,
              format.sampleRate == Double(chunk.sampleRate),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(chunk.samples.count)),
              let channel = buffer.floatChannelData?[0] else {
            throw PlaybackError.invalidBuffer
        }
        buffer.frameLength = AVAudioFrameCount(chunk.samples.count)
        chunk.samples.withUnsafeBufferPointer { source in
            channel.update(from: source.baseAddress!, count: source.count)
        }
        // If inference starved playback, the node's clock may have passed the previous end position.
        scheduledFrames = max(scheduledFrames, playedFrames) + Int64(chunk.samples.count)
        pendingBuffers += 1
        node.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.generation == current else { return }
                self.pendingBuffers = max(0, self.pendingBuffers - 1)
            }
        }
        if !node.isPlaying { node.play() }
    }

    func finish() async throws {
        let current = generation
        while pendingBuffers > 0 {
            try await Task.sleep(for: .milliseconds(20))
            guard generation == current else { throw CancellationError() }
        }
        stop()
    }

    func stop() {
        generation = UUID()
        node?.stop()
        engine?.stop()
        node = nil
        engine = nil
        format = nil
        scheduledFrames = 0
        pendingBuffers = 0
        #if os(iOS)
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif
    }

    private func start(sampleRate: Int) throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setPreferredIOBufferDuration(0.01)
        try session.setActive(true)
        #endif
        let engine = AVAudioEngine()
        let node = AVAudioPlayerNode()
        guard let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1) else {
            throw PlaybackError.invalidBuffer
        }
        engine.attach(node)
        // The mixer converts the model's 24 kHz mono PCM to the current hardware output format.
        engine.connect(node, to: engine.mainMixerNode, format: format)
        engine.prepare()
        try engine.start()
        self.engine = engine
        self.node = node
        self.format = format
    }
}

private enum PlaybackError: LocalizedError {
    case invalidBuffer
    var errorDescription: String? { "Ses tamponu oluşturulamadı." }
}
