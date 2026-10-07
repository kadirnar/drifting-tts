import Combine
import DriftingTTS
import Foundation

@MainActor
final class SpeechViewModel: ObservableObject {
    @Published var text = "Merhaba! Bu ses, iPhone üzerinde çevrimdışı olarak üretildi."
    @Published var voice = "studio"
    @Published private(set) var isBusy = false
    @Published private(set) var isReady = false
    @Published private(set) var isDownloading = false
    @Published private(set) var progress = 0.0
    @Published private(set) var status = "Modeli bir kez indirin; ardından internet bağlantısı gerekmez."
    @Published private(set) var error: String?
    @Published private(set) var firstPCMSeconds: Double?
    @Published private(set) var scheduledAudioSeconds: Double?
    @Published private(set) var audioSeconds = 0.0
    @Published private(set) var peakMemoryBytes = 0
    @Published private(set) var synthesisSeconds: Double?

    private let store = ModelStore()
    private let player = PCMPlayer()
    private var synthesizer: DriftingSynthesizer?
    private var operation: Task<Void, Never>?
    private var requestID = UUID()
    private var unloadAfterOperation = false

    func prepare() {
        guard !isBusy else { return }
        #if targetEnvironment(simulator)
        error = "MLX için fiziksel iPhone gerekir. Simülatörde ses üretimi desteklenmez."
        return
        #else
        isBusy = true
        isDownloading = true
        error = nil
        progress = 0
        status = "Model dosyaları kontrol ediliyor…"
        let request = UUID()
        requestID = request
        operation = Task {
            defer { completeOperation() }
            do {
                let directory = try await store.prepare { [weak self] value in
                    await MainActor.run {
                        guard let self, self.requestID == request, self.isBusy, self.isDownloading else { return }
                        self.progress = max(self.progress, value)
                        self.status = self.progress < 1 ? "Model indiriliyor ve doğrulanıyor…" : "Model yükleniyor…"
                    }
                }
                try Task.checkCancellation()
                // Loading safetensors is substantial work: keep it outside the main actor.
                let loading = Task.detached(priority: .userInitiated) {
                    try DriftingSynthesizer(modelDirectory: directory)
                }
                let loaded = try await withTaskCancellationHandler {
                    try await loading.value
                } onCancel: {
                    loading.cancel()
                }
                try Task.checkCancellation()
                synthesizer = loaded
                isReady = true
                status = "Çevrimdışı hazır"
            } catch is CancellationError {
                status = "Hazırlama durduruldu."
            } catch {
                if Task.isCancelled {
                    status = "Hazırlama durduruldu."
                } else {
                    self.error = error.localizedDescription
                    status = "Model hazırlanamadı."
                }
            }
        }
        #endif
    }

    func speak() {
        guard !isBusy, let synthesizer, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        isBusy = true
        error = nil
        firstPCMSeconds = nil
        scheduledAudioSeconds = nil
        synthesisSeconds = nil
        audioSeconds = 0
        peakMemoryBytes = 0
        status = "Ses üretiliyor…"
        let request = UUID()
        requestID = request
        let input = text
        var options = SynthesisOptions()
        options.voice = voice
        options.chunkFrames = 128
        options.firstChunkFrames = 24
        let started = ProcessInfo.processInfo.systemUptime
        operation = Task {
            defer { completeOperation() }
            do {
                let metrics = try await synthesizer.synthesize(input, options: options) { [weak self] chunk in
                    try await self?.receive(chunk, request: request, started: started)
                }
                try Task.checkCancellation()
                peakMemoryBytes = metrics.peakMemoryBytes
                synthesisSeconds = metrics.totalSeconds
                if firstPCMSeconds == nil { firstPCMSeconds = metrics.ttfaSeconds }
                status = "Ses çalınıyor…"
                try await player.finish()
                status = "Tamamlandı · çevrimdışı"
            } catch is CancellationError {
                player.stop()
                status = "Durduruldu."
            } catch {
                player.stop()
                self.error = error.localizedDescription
                status = "Ses üretilemedi."
            }
        }
    }

    private func receive(_ chunk: AudioChunk, request: UUID, started: Double) async throws {
        try Task.checkCancellation()
        guard requestID == request else { throw CancellationError() }
        if firstPCMSeconds == nil, !chunk.isSilence, !chunk.samples.isEmpty {
            firstPCMSeconds = ProcessInfo.processInfo.systemUptime - started
        }
        try await player.enqueue(chunk)
        if scheduledAudioSeconds == nil, !chunk.isSilence, !chunk.samples.isEmpty {
            scheduledAudioSeconds = ProcessInfo.processInfo.systemUptime - started
        }
        audioSeconds += Double(chunk.samples.count) / Double(chunk.sampleRate)
        status = "Ses üretiliyor ve çalınıyor…"
    }

    func cancel() {
        guard isBusy else { return }
        requestID = UUID()
        operation?.cancel()
        player.stop()
        status = "Durduruluyor…"
    }

    func memoryWarning() {
        unloadAfterOperation = true
        cancel()
        if !isBusy { releaseModel() }
    }

    private func completeOperation() {
        operation = nil
        isBusy = false
        isDownloading = false
        if unloadAfterOperation { releaseModel() }
    }

    private func releaseModel() {
        synthesizer = nil
        isReady = false
        unloadAfterOperation = false
        status = "Bellek baskısı nedeniyle model serbest bırakıldı. İndirilen dosyalar korunuyor."
    }
}
