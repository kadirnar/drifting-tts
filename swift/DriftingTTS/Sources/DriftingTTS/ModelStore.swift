import CryptoKit
import Foundation

/// Downloads the exact validated checkpoint once; subsequent calls work without a network connection.
public actor ModelStore {
    public static let revision = "2de3308045f6f559b2efa2d8cca3749fa3262848"
    public static let downloadBytes: Int64 = 495_773_133

    private struct ModelFile: Sendable {
        let name: String
        let bytes: Int64
        let sha256: String
    }

    private static let files = [
        ModelFile(name: "config.json", bytes: 1_376,
                  sha256: "e3a4e05e8354770e45282a26aab84bd7421b7313214498ba71cee7a3935f2cb4"),
        ModelFile(name: "model.safetensors", bytes: 270_789_403,
                  sha256: "f462bd0647bd6b75773968c7dd243f45882b67383233c64fa61c7dc7cd28d32f"),
        ModelFile(name: "vocoder.safetensors", bytes: 224_982_354,
                  sha256: "c1dc7b0b731b9ca84b571072cdbf4e68131bc5d199a9dce93d726d55af20c70f"),
    ]

    private let root: URL
    private var pending: Task<URL, Error>?

    public init(directory: URL? = nil) {
        root = directory ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("DriftingTTS/Models", isDirectory: true)
    }

    /// Progress covers download and verification. A complete cache is verified before it is returned.
    public func prepare(onProgress: @escaping @Sendable (Double) async -> Void = { _ in }) async throws -> URL {
        if let pending { return try await pending.value }
        let task = Task { try await install(onProgress: onProgress) }
        pending = task
        defer { pending = nil }
        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    private func install(onProgress: @escaping @Sendable (Double) async -> Void) async throws -> URL {
        let manager = FileManager.default
        let destination = root.appendingPathComponent(Self.revision, isDirectory: true)
        try manager.createDirectory(at: root, withIntermediateDirectories: true)
        var excludedRoot = root
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try excludedRoot.setResourceValues(values)
        try Task.checkCancellation()
        if try Self.isValid(directory: destination) {
            await onProgress(1)
            return destination
        }

        let staging = root.appendingPathComponent(".\(UUID().uuidString).partial", isDirectory: true)
        try manager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: staging) }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 60 * 30
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        var downloaded: Int64 = 0
        await onProgress(0)
        for file in Self.files {
            try Task.checkCancellation()
            let address = "https://huggingface.co/Vyvo/drifting-tts-tr/resolve/\(Self.revision)/mlx/\(file.name)"
            guard let url = URL(string: address) else { throw StoreError.invalidURL }
            let delegate = DownloadProgress(offset: downloaded, expected: file.bytes,
                                            total: Self.downloadBytes, callback: onProgress)
            let (temporary, response) = try await session.download(from: url, delegate: delegate)
            defer { try? manager.removeItem(at: temporary) }
            guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
                throw StoreError.download(file.name)
            }
            let target = staging.appendingPathComponent(file.name)
            try manager.moveItem(at: temporary, to: target)
            guard try Self.verify(file: target, expected: file) else { throw StoreError.integrity(file.name) }
            downloaded += file.bytes
            await onProgress(min(0.99, Double(downloaded) / Double(Self.downloadBytes)))
        }
        try Task.checkCancellation()
        // Publish the entire verified directory at once. Preserve an older directory until replacement succeeds.
        if manager.fileExists(atPath: destination.path) {
            _ = try manager.replaceItemAt(destination, withItemAt: staging)
        } else {
            try manager.moveItem(at: staging, to: destination)
        }
        await onProgress(1)
        return destination
    }

    private static func isValid(directory: URL) throws -> Bool {
        for file in files {
            guard try verify(file: directory.appendingPathComponent(file.name), expected: file) else { return false }
        }
        return true
    }

    private static func verify(file: URL, expected: ModelFile) throws -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: file.path),
              let size = attributes[.size] as? NSNumber, size.int64Value == expected.bytes else { return false }
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hash = SHA256()
        while let bytes = try handle.read(upToCount: 1_048_576), !bytes.isEmpty {
            try Task.checkCancellation()
            hash.update(data: bytes)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined() == expected.sha256
    }
}

private enum StoreError: LocalizedError {
    case invalidURL
    case download(String)
    case integrity(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL: "The pinned model download URL is invalid."
        case .download(let name): "Could not download \(name). Please check the network connection."
        case .integrity(let name): "The downloaded \(name) failed its size or SHA-256 check. Please retry."
        }
    }
}

private final class DownloadProgress: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let offset: Int64
    private let expected: Int64
    private let total: Int64
    private let callback: @Sendable (Double) async -> Void
    private let lock = NSLock()
    private var lastReported = -1

    init(offset: Int64, expected: Int64, total: Int64, callback: @escaping @Sendable (Double) async -> Void) {
        self.offset = offset
        self.expected = expected
        self.total = total
        self.callback = callback
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let progress = min(0.99, Double(offset + min(totalBytesWritten, expected)) / Double(total))
        let percent = Int(progress * 100)
        lock.lock()
        let changed = percent > lastReported
        if changed { lastReported = percent }
        lock.unlock()
        if changed {
            let callback = callback
            Task { await callback(progress) }
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {}
}
