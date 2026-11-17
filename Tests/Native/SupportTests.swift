import XCTest
@testable import KmzMacSupport

final class SupportTests: XCTestCase {
    func testCoMetIsAnIndependentSuiteTool() {
        XCTAssertEqual(SuiteTool.allCases.count, 4)
        XCTAssertEqual(SuiteTool.comet.rawValue, "KmzCoMet")
        XCTAssertEqual(SuiteTool.comet.symbol, "tablecells")
        XCTAssertEqual(SuiteTool.comet.application(in: URL(fileURLWithPath: "/fixture")).lastPathComponent, "KmzCoMet.app")
    }
    func testPortableToolLookupDoesNotFallBackToSiblings() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        for embedded in [false, true] {
            let app = root.appendingPathComponent(embedded ? "Portable.app" : "Sibling.app")
            let contents = app.appendingPathComponent("Contents")
            try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
            let info: [String: Any] = ["CFBundleIdentifier": "test." + UUID().uuidString,
                                      "CFBundlePackageType": "APPL", "KMZContainsTools": embedded]
            try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
                .write(to: contents.appendingPathComponent("Info.plist"))
            let bundle = try XCTUnwrap(Bundle(url: app))
            XCTAssertEqual(NativeSupport.toolDirectory(for: bundle).standardizedFileURL,
                           (embedded ? contents.appendingPathComponent("Helpers", isDirectory: true) : root).standardizedFileURL)
        }
    }
    func testOutputRepresentationNamesAndNormalization() {
        let folder = URL(fileURLWithPath: "/fixture", isDirectory: true)
        for name in ["sample", "sample.mzML", "sample.sp.mzML", "sample.SP.MZML", "sample.mzML.sp.mzML"] {
            XCTAssertEqual(OutputProfile.sparse.outputURL(folder.appendingPathComponent(name)).lastPathComponent, "sample.sp.mzML")
            XCTAssertEqual(OutputProfile.full.outputURL(folder.appendingPathComponent(name)).lastPathComponent, "sample.mzML")
        }
        XCTAssertEqual(OutputProfile.full.rawValue, "full")
        XCTAssertEqual(OutputProfile.sparse.rawValue, "sparse")
        XCTAssertEqual(OutputProfile.sparse.title, "Sparse Profile v3")
        XCTAssertEqual(OutputProfile.full.suggestedOutputURL(for: folder.appendingPathComponent("sample.sp.wiff")).lastPathComponent, "sample.sp.mzML")
        XCTAssertEqual(OutputProfile.sparse.suggestedOutputURL(for: folder.appendingPathComponent("sample.sp.wiff")).lastPathComponent, "sample.sp.sp.mzML")
    }
    func testProgressUsesLatestCompleteUpdateAndRejectsInvalidCounts() {
        let progress = ConversionProgress.parse("warning\rConverting 1/500 spectra\rConverting 251/500 spectra")
        XCTAssertEqual(progress?.done, 251)
        XCTAssertEqual(progress?.total, 500)
        XCTAssertNil(ConversionProgress.parse("Converting 1/0 spectra"))
        XCTAssertNil(ConversionProgress.parse("Converting 8/2 spectra"))
        XCTAssertNil(ConversionProgress.parse("Reading acquisition"))
    }
    func testSiblingAvailabilityRequiresAnExecutable() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let executable = SuiteTool.viewer.application(in: directory).appendingPathComponent("Contents/MacOS/KmzViewer")
        try FileManager.default.createDirectory(at: executable.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("fixture".utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: executable.path)
        XCTAssertFalse(SuiteTool.viewer.isAvailable(in: directory))
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        XCTAssertTrue(SuiteTool.viewer.isAvailable(in: directory))
        XCTAssertFalse(SuiteTool.viewer2.isAvailable(in: directory))
    }
}
