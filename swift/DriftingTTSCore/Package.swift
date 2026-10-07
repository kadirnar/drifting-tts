// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DriftingTTSCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "DriftingTTSCore", targets: ["DriftingTTSCore"]),
        .executable(name: "drifting-core-check", targets: ["DriftingCoreCheck"]),
    ],
    targets: [
        .target(name: "DriftingTTSCore", resources: [.copy("Resources/TurkishFrontend.js")],
                linkerSettings: [.linkedFramework("JavaScriptCore")]),
        .testTarget(name: "DriftingTTSCoreTests", dependencies: ["DriftingTTSCore"],
                    resources: [.copy("Resources/text_fixtures.json")]),
        // Foundation-only checks are runnable with Command Line Tools, which do not ship XCTest.
        .executableTarget(name: "DriftingCoreCheck", dependencies: ["DriftingTTSCore"], path: "Checks"),
    ]
)
