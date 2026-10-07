// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "DriftingTTS",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "DriftingTTS", targets: ["DriftingTTS"]),
        .executable(name: "drifting-tts-swift", targets: ["DriftingTTSCLI"]),
    ],
    dependencies: [
        .package(path: "../DriftingTTSCore"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.32.3"),
    ],
    targets: [
        .target(name: "DriftingTTS", dependencies: [
            .product(name: "DriftingTTSCore", package: "DriftingTTSCore"),
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
            .product(name: "MLXRandom", package: "mlx-swift"),
        ]),
        .executableTarget(name: "DriftingTTSCLI", dependencies: ["DriftingTTS"]),
        .testTarget(name: "DriftingTTSTests", dependencies: ["DriftingTTS"], resources: [.copy("Fixtures")]),
    ]
)
