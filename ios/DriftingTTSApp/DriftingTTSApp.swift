import AVFoundation
import SwiftUI

@main
struct DriftingTTSApp: App {
    var body: some Scene {
        WindowGroup { SpeechView() }
    }
}

private struct SpeechView: View {
    @StateObject private var model = SpeechViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Cihaz üzerinde Türkçe ses", systemImage: "waveform")
                            .font(.title2.bold())
                        Text("Metniniz telefonda kalır. İlk indirmeden sonra ses üretimi çevrimdışı çalışır.")
                            .foregroundStyle(.secondary)
                    }

                    GroupBox {
                        VStack(alignment: .leading, spacing: 12) {
                            Label(model.isReady ? "Model hazır" : "Model hazırlığı",
                                  systemImage: model.isReady ? "checkmark.circle.fill" : "arrow.down.circle")
                                .font(.headline)
                                .foregroundStyle(model.isReady ? .green : .primary)
                            Text(model.status).font(.subheadline).foregroundStyle(.secondary)
                            if model.isDownloading { ProgressView(value: model.progress) }
                            if !model.isReady {
                                Button("Modeli hazırla · 496 MB") { model.prepare() }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(model.isBusy)
                            }
                        }.frame(maxWidth: .infinity, alignment: .leading)
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Okunacak metin").font(.headline)
                        TextEditor(text: $model.text)
                            .frame(minHeight: 150)
                            .padding(8)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
                            .disabled(model.isBusy)
                            .accessibilityLabel("Okunacak Türkçe metin")
                        Picker("Ses", selection: $model.voice) {
                            Text("Stüdyo").tag("studio")
                            Text("Erkek").tag("male")
                            Text("Kadın").tag("female")
                        }.pickerStyle(.segmented).disabled(model.isBusy)

                        HStack {
                            Button {
                                model.speak()
                            } label: {
                                Label("Seslendir", systemImage: "play.fill")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .disabled(!model.isReady || model.isBusy || model.text.isEmpty)
                            Button("Durdur", role: .destructive) { model.cancel() }
                                .buttonStyle(.bordered)
                                .disabled(!model.isBusy)
                        }
                    }

                    if model.firstPCMSeconds != nil {
                        GroupBox("Bu çalıştırma") {
                            VStack(spacing: 10) {
                                metric("İlk ses verisi · TTFA", value: milliseconds(model.firstPCMSeconds))
                                metric("İlk oynatma kuyruğu", value: milliseconds(model.scheduledAudioSeconds))
                                metric("Üretilen ses", value: String(format: "%.2f sn", model.audioSeconds))
                                if let total = model.synthesisSeconds {
                                    metric("Üretim ve akış", value: String(format: "%.2f sn", total))
                                }
                                if model.peakMemoryBytes > 0 {
                                    metric("MLX tepe · oturum",
                                           value: String(format: "%.0f MB", Double(model.peakMemoryBytes) / 1_000_000))
                                }
                                Text("TTFA, ilk PCM verisinin hazır olduğu ana kadardır. Hoparlör gecikmesini içermez.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }.padding(.top, 8)
                        }
                    }
                    if let error = model.error {
                        Label(error, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red).font(.subheadline)
                    }
                    Text("En doğru hız ölçümü için fiziksel cihazda Release yapılandırmasını kullanın. "
                         + "Ölçümler bu cihazdaki çalıştırmaya aittir.")
                        .font(.caption).foregroundStyle(.secondary)
                }.padding()
            }
            .navigationTitle("Drifting TTS")
            .onChange(of: scenePhase) { _, phase in
                if phase != .active { model.cancel() }
            }
            #if os(iOS)
            .onReceive(NotificationCenter.default.publisher(for: UIApplication.didReceiveMemoryWarningNotification)) { _ in
                model.memoryWarning()
            }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.interruptionNotification)) { _ in
                model.cancel()
            }
            #endif
        }
    }

    private func metric(_ label: String, value: String) -> some View {
        HStack {
            Text(label).foregroundStyle(.secondary)
            Spacer()
            Text(value).monospacedDigit().bold()
        }.font(.subheadline)
    }

    private func milliseconds(_ seconds: Double?) -> String {
        seconds.map { String(format: "%.0f ms", $0 * 1000) } ?? "—"
    }
}
