import Foundation
import XCTest
@testable import DriftingTTSCore

final class TurkishFrontendTests: XCTestCase {
    private struct Fixtures: Decodable {
        let symbols: [String]
        let symbol_to_id: [String: Int]
        let pad_id, blank_id: Int
        let number_to_words, ordinal_to_words: [[String]]
        let cases: [Case]
        struct Case: Decodable {
            let input, normalized, source: String
            let ids: [Int]
            let split, split_short, split_raw: [String]
        }
    }

    func testCompletePythonFixtureParity() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "text_fixtures", withExtension: "json"))
        let fixture = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: url))
        let frontend = try TurkishFrontend()
        XCTAssertEqual(frontend.symbols, fixture.symbols)
        XCTAssertEqual(Dictionary(uniqueKeysWithValues: frontend.symbols.enumerated().map { ($0.element, $0.offset) }),
                       fixture.symbol_to_id)
        XCTAssertEqual(frontend.padID, fixture.pad_id)
        XCTAssertEqual(frontend.blankID, fixture.blank_id)
        for pair in fixture.number_to_words {
            XCTAssertEqual(try frontend.numberToWords(pair[0]), pair[1], pair[0])
        }
        for pair in fixture.ordinal_to_words {
            XCTAssertEqual(try frontend.ordinalToWords(pair[0]), pair[1], pair[0])
        }
        XCTAssertEqual(fixture.cases.count, 1_240)
        for item in fixture.cases {
            let label = "\(item.source): \(item.input)"
            XCTAssertEqual(try frontend.normalize(item.input), item.normalized, label)
            XCTAssertEqual(try frontend.textToIDs(item.input), item.ids, label)
            XCTAssertEqual(try frontend.textToIDs(item.input, intersperseBlank: false),
                           item.ids.enumerated().filter { $0.offset % 2 == 1 }.map(\.element), label)
            XCTAssertEqual(try frontend.textToIDs(item.normalized, normalized: true), item.ids, label)
            XCTAssertEqual(try frontend.splitSentences(item.normalized), item.split, label)
            XCTAssertEqual(try frontend.splitSentences(item.normalized, maxChars: 40), item.split_short, label)
            XCTAssertEqual(try frontend.splitSentences(item.input, maxChars: 30), item.split_raw, label)
        }
    }

    func testSupplementaryDigitsAndApostrophesInJavaScriptCore() throws {
        let frontend = try TurkishFrontend()
        XCTAssertEqual(try frontend.normalize("𝟓:𝟓"), "beş beş")
        XCTAssertEqual(try frontend.normalize("𝟏𝟐'nın"), "on ikinın")
    }

    func testInputIsBridgedWithoutScriptInterpolation() throws {
        let frontend = try TurkishFrontend()
        let text = "\"); throw new Error('injected'); //"
        XCTAssertNoThrow(try frontend.normalize(text))
        XCTAssertEqual(try frontend.normalize("Merhaba"), "merhaba")
    }

    func testEmptyInputAndValidation() throws {
        let frontend = try TurkishFrontend()
        XCTAssertEqual(try frontend.normalize(""), "")
        XCTAssertEqual(try frontend.textToIDs(""), [frontend.blankID])
        XCTAssertEqual(try frontend.splitSentences(""), [])
        XCTAssertThrowsError(try frontend.splitSentences("merhaba", maxChars: 0))
        XCTAssertThrowsError(try frontend.splitSentences("merhaba", maxChars: -1))
        XCTAssertThrowsError(try frontend.textToIDs("😃", normalized: true))
        // A failed invocation must not poison subsequent calls on this context.
        XCTAssertEqual(try frontend.normalize("İYİ"), "iyi")
    }
}
