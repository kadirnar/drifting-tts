import Foundation
import DriftingTTSCore

struct CheckFailure: Error, CustomStringConvertible {
    let description: String
}

/// A Command Line Tools fallback for hosts without XCTest; consumes the same Python-generated fixtures.
@main struct CoreCheck {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else {
            throw CheckFailure(description: "Usage: drifting-core-check <web/tests/text_fixtures.json>")
        }
        let fixtureURL = URL(fileURLWithPath: CommandLine.arguments[1])
        guard let fixture = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as? [String: Any],
              let cases = fixture["cases"] as? [[String: Any]],
              let numbers = fixture["number_to_words"] as? [[String]],
              let ordinals = fixture["ordinal_to_words"] as? [[String]] else {
            throw CheckFailure(description: "Invalid fixture JSON")
        }
        var checks = 0
        func check(_ actual: Any, _ expected: Any, _ name: String) throws {
            checks += 1
            guard let a = actual as? NSObject, let e = expected as? NSObject, a.isEqual(e) else {
                throw CheckFailure(description: "\(name): \(actual) != \(expected)")
            }
        }
        func rejects(_ name: String, _ body: () throws -> Void) throws {
            checks += 1
            do { try body() } catch { return }
            throw CheckFailure(description: "Expected rejection: \(name)")
        }
        let frontend = try TurkishFrontend()
        try check(frontend.symbols, fixture["symbols"]!, "symbols")
        try check(Dictionary(uniqueKeysWithValues: frontend.symbols.enumerated().map { ($0.element, $0.offset) }),
                  fixture["symbol_to_id"]!, "symbol_to_id")
        try check(frontend.padID, fixture["pad_id"]!, "pad_id")
        try check(frontend.blankID, fixture["blank_id"]!, "blank_id")
        for pair in numbers { try check(frontend.numberToWords(pair[0]), pair[1], "number \(pair[0])") }
        for pair in ordinals { try check(frontend.ordinalToWords(pair[0]), pair[1], "ordinal \(pair[0])") }
        for item in cases {
            let input = item["input"] as! String, normalized = item["normalized"] as! String
            let ids = item["ids"] as! [Int]
            try check(frontend.normalize(input), normalized, "normalize \(input)")
            try check(frontend.textToIDs(input), ids, "IDs \(input)")
            try check(frontend.textToIDs(input, intersperseBlank: false),
                      ids.enumerated().filter { $0.offset % 2 == 1 }.map(\.element), "plain IDs \(input)")
            try check(frontend.textToIDs(normalized, normalized: true), ids, "normalized IDs \(input)")
            try check(frontend.splitSentences(normalized), item["split"]!, "split \(input)")
            try check(frontend.splitSentences(normalized, maxChars: 40), item["split_short"]!, "split 40 \(input)")
            try check(frontend.splitSentences(input, maxChars: 30), item["split_raw"]!, "raw split 30 \(input)")
        }
        try check(frontend.normalize("𝟓:𝟓"), "beş beş", "supplementary digits")
        try check(frontend.normalize("𝟏𝟐'nın"), "on ikinın", "supplementary suffix")
        _ = try frontend.normalize("\"); throw new Error('injected'); //")
        try check(frontend.normalize("Merhaba"), "merhaba", "bridged input")
        try check(frontend.normalize(""), "", "empty normalize")
        try check(frontend.textToIDs(""), [frontend.blankID], "empty IDs")
        try check(frontend.splitSentences(""), [String](), "empty split")
        try rejects("zero maxChars") { _ = try frontend.splitSentences("merhaba", maxChars: 0) }
        try rejects("negative maxChars") { _ = try frontend.splitSentences("merhaba", maxChars: -1) }
        try rejects("invalid normalized symbol") { _ = try frontend.textToIDs("😃", normalized: true) }
        try check(frontend.normalize("İYİ"), "iyi", "context recovers after error")

        let configJSON = #"""
        {"model":{"text":{"d":8,"heads":2,"layers":1,"ffn":16,"spk_dim":4},
          "gen":{"hidden":8,"depth":1,"heads":2,"patch":2,"mlp_ratio":3,
          "n_registers":2,"noise_classes":4,"noise_coords":2}},
          "vocoder":{"num_mels":4,"upsample_rates":[2,2],"upsample_kernel_sizes":[4,4],
          "upsample_initial_channel":16,"resblock_kernel_sizes":[3],"resblock_dilation_sizes":[[1,3,5]],
          "activation":"snakebeta","snake_logscale":true},
          "num_speakers":2,"n_vocab":39,"n_mels":4,"sample_rate":24000,"hop_length":4,
          "stats":{"mean":-5,"std":2},"duration_scales":{"1":1.25},
          "voices":{"Deniz":0,"Ece":1},"default_voice":"Deniz"}
        """#
        func decode(_ mutate: ((inout [String: Any]) -> Void)? = nil) throws -> CheckpointConfig {
            var object = try JSONSerialization.jsonObject(with: Data(configJSON.utf8)) as! [String: Any]
            mutate?(&object)
            return try JSONDecoder().decode(CheckpointConfig.self, from: JSONSerialization.data(withJSONObject: object))
        }
        let config = try decode()
        try check(config.model.text.spkDim, 4, "spk_dim")
        try check(config.model.gen.numSteps, 1, "num_steps default")
        try check(config.model.gen.residualPrior, true, "residual_prior default")
        try check(config.model.pitch.enabled, false, "pitch default")
        try check(config.vocoder.useBiasAtFinal && config.vocoder.useTanhAtFinal, true, "vocoder defaults")
        try check(config.speakerID(), 0, "default voice")
        try check(config.speakerID("Ece"), 1, "named voice")
        try check(config.speakerID("1"), 1, "numeric voice")
        try check(config.durationScale(for: 1), Float(1.25), "duration scale")
        try rejects("unknown voice") { _ = try config.speakerID("missing") }
        try rejects("out of range speaker") { _ = try config.speakerID("2") }
        let encoded = try JSONEncoder().encode(config)
        let roundTrip = try JSONDecoder().decode(CheckpointConfig.self, from: encoded)
        try check(roundTrip.sampleRate, 24000, "config round trip")
        for (key, value) in [("n_vocab", 40), ("n_mels", 5), ("hop_length", 8), ("num_speakers", 0)] {
            try rejects(key) { _ = try decode { $0[key] = value } }
        }
        try rejects("default voice") { _ = try decode { $0["default_voice"] = "unknown" } }
        try rejects("duration scale") { _ = try decode { $0["duration_scale"] = 0 } }
        try rejects("speaker duration scale") { _ = try decode { $0["duration_scales"] = ["2": 1.0] } }
        try rejects("temperature") { _ = try decode { $0["temperature"] = -1 } }
        try rejects("normalization") { _ = try decode { $0["stats"] = ["mean": 0, "std": 0] } }
        for (key, value) in [("num_steps", 2), ("hidden", 6), ("heads", 0), ("patch", 0)] {
            try rejects(key) {
                _ = try decode { object in
                    var model = object["model"] as! [String: Any], generator = model["gen"] as! [String: Any]
                    generator[key] = value; model["gen"] = generator; object["model"] = model
                }
            }
        }
        print("DriftingTTSCore: \(checks) checks passed across \(cases.count) frontend fixtures and config validation")
    }
}
