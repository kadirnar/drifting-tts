import Foundation
import XCTest
@testable import DriftingTTSCore

final class ConfigurationTests: XCTestCase {
    private let validJSON = #"""
    {
      "model": {
        "text": {"d": 8, "heads": 2, "layers": 1, "ffn": 16, "spk_dim": 4},
        "gen": {"hidden": 8, "depth": 1, "heads": 2, "patch": 2, "mlp_ratio": 3.0,
                "n_registers": 2, "noise_classes": 4, "noise_coords": 2}
      },
      "vocoder": {"num_mels": 4, "upsample_rates": [2, 2], "upsample_kernel_sizes": [4, 4],
        "upsample_initial_channel": 16, "resblock_kernel_sizes": [3],
        "resblock_dilation_sizes": [[1, 3, 5]], "activation": "snakebeta", "snake_logscale": true},
      "num_speakers": 2, "n_vocab": 39, "n_mels": 4, "sample_rate": 24000, "hop_length": 4,
      "stats": {"mean": -5.0, "std": 2.0}, "duration_scales": {"1": 1.25},
      "voices": {"Deniz": 0, "Ece": 1}, "default_voice": "Deniz"
    }
    """#

    private func decode(_ mutate: ((inout [String: Any]) -> Void)? = nil) throws -> CheckpointConfig {
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(validJSON.utf8)) as? [String: Any])
        mutate?(&object)
        return try JSONDecoder().decode(CheckpointConfig.self, from: JSONSerialization.data(withJSONObject: object))
    }

    func testSnakeCaseFieldsAndLegacyDefaults() throws {
        let config = try decode()
        XCTAssertEqual(config.model.text.spkDim, 4)
        XCTAssertEqual(config.model.gen.mlpRatio, 3)
        XCTAssertEqual(config.model.gen.nRegisters, 2)
        XCTAssertEqual(config.model.gen.numSteps, 1)
        XCTAssertTrue(config.model.gen.residualPrior)
        XCTAssertFalse(config.model.pitch.enabled)
        XCTAssertEqual(config.vocoder.resblock, "1")
        XCTAssertTrue(config.vocoder.useTanhAtFinal)
        XCTAssertTrue(config.vocoder.useBiasAtFinal)
        XCTAssertEqual(config.durationScale, 1)
        XCTAssertEqual(config.temperature, 0.3)
        XCTAssertEqual(try config.speakerID(), 0)
        XCTAssertEqual(try config.speakerID("Ece"), 1)
        XCTAssertEqual(try config.speakerID("1"), 1)
        XCTAssertEqual(config.durationScale(for: 0), 1)
        XCTAssertEqual(config.durationScale(for: 1), 1.25)
        XCTAssertThrowsError(try config.speakerID("missing"))
        XCTAssertThrowsError(try config.speakerID("-1"))
        XCTAssertThrowsError(try config.speakerID("2"))
    }

    func testRoundTripPreservesExplicitFields() throws {
        let config = try decode { object in
            var model = object["model"] as! [String: Any]
            model["pitch"] = ["enabled": true]
            var generator = model["gen"] as! [String: Any]
            generator["residual_prior"] = false
            generator["num_steps"] = 1
            model["gen"] = generator
            object["model"] = model
            var vocoder = object["vocoder"] as! [String: Any]
            vocoder["use_tanh_at_final"] = false
            vocoder["use_bias_at_final"] = false
            object["vocoder"] = vocoder
        }
        let data = try JSONEncoder().encode(config)
        let roundTrip = try JSONDecoder().decode(CheckpointConfig.self, from: data)
        XCTAssertTrue(roundTrip.model.pitch.enabled)
        XCTAssertFalse(roundTrip.model.gen.residualPrior)
        XCTAssertFalse(roundTrip.vocoder.useTanhAtFinal)
        XCTAssertFalse(roundTrip.vocoder.useBiasAtFinal)
    }

    func testUnsupportedCheckpointsFailAtLoadTime() throws {
        for (key, value) in [("n_vocab", 40), ("n_mels", 5), ("hop_length", 8), ("num_speakers", 0)] {
            XCTAssertThrowsError(try decode { $0[key] = value }, key)
        }
        XCTAssertThrowsError(try decode { $0["default_voice"] = "unknown" })
        XCTAssertThrowsError(try decode { $0["duration_scale"] = 0 })
        XCTAssertThrowsError(try decode { $0["duration_scales"] = ["2": 1.0] })
        XCTAssertThrowsError(try decode { $0["temperature"] = -1 })
        XCTAssertThrowsError(try decode { $0["stats"] = ["mean": 0, "std": 0] })
        for (key, value) in [("num_steps", 2), ("hidden", 6), ("heads", 0), ("patch", 0)] {
            XCTAssertThrowsError(try decode { object in
                var model = object["model"] as! [String: Any]
                var generator = model["gen"] as! [String: Any]
                generator[key] = value
                model["gen"] = generator
                object["model"] = model
            }, key)
        }
        XCTAssertThrowsError(try decode { object in
            var vocoder = object["vocoder"] as! [String: Any]
            vocoder["upsample_kernel_sizes"] = [3, 4]
            object["vocoder"] = vocoder
        })
    }

    func testConfigLoadsFromLocalFile() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data(validJSON.utf8).write(to: url)
        XCTAssertEqual(try CheckpointConfig.load(from: url).sampleRate, 24_000)
    }
}
