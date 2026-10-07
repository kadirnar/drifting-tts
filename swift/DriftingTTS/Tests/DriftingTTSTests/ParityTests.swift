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

    func testCancellationInvalidOptionsAndEngineReuse() async throws {
        let errors = try await Verification.runLifecycle(directory: fixtures)
        XCTAssertEqual(errors.count, 4)
        XCTAssertEqual(errors["lifecycle.pre_cancelled_load"], 0)
        XCTAssertEqual(errors["lifecycle.invalid_options"], 0)
        XCTAssertLessThan(try XCTUnwrap(errors["lifecycle.cancelled_prefix"]), 1e-4)
        XCTAssertLessThan(try XCTUnwrap(errors["lifecycle.reused_waveform"]), 1e-4)
    }
}
