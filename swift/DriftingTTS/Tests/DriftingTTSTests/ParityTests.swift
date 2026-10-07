import Foundation
import XCTest
@testable import DriftingTTS

final class ParityTests: XCTestCase {
    private var fixtures: URL {
        get throws { try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil)) }
    }

    func testAcousticVocoderAndRandomParity() throws {
        let errors = try Verification.run(directory: fixtures)
        XCTAssertEqual(errors.count, 15)
        XCTAssertTrue(errors.values.allSatisfy { $0.isFinite && $0 < 1e-4 })
    }

    func testPublicStreamingPipelineParity() async throws {
        let errors = try await Verification.runPipeline(directory: fixtures)
        XCTAssertLessThan(try XCTUnwrap(errors["pipeline.waveform"]), 1e-4)
    }
}
