import XCTest
import AppKit
import KmzMacSupport
@testable import KmzConvert

@MainActor final class ConversionModelTests: XCTestCase {
    private var folder: URL!

    override func setUp() async throws {
        folder = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".local/convert-tests/" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        try FileManager.default.removeItem(at: folder)
    }

    private func input(_ name: String) -> URL { folder.appendingPathComponent(name) }

    private func converter(_ body: String = #"""
        set -C
        printf '%s\n' "$1|$2|$3|$4|$5" >> started.txt
        mkdir active || exit 9
        trap 'rmdir active' EXIT
        case "$1" in
          *fail.wiff) echo 'kmzconvert: fixture file failed'; exit 1 ;;
        esac
        printf '\rConverting 1/4 spectra'
        sleep 0.05
        printf '\rConverting 4/4 spectra\n'
        if [ "$2" = '--inspect' ]; then echo '{"spectra":4}'; else
          printf 'fixture:%s' "$5" > "$3" || exit 1
          printf 'Wrote %s\n' "$3"
        fi
        """#) throws -> URL {
        let path = folder.appendingPathComponent("converter")
        let discovery = #"""
        if [ "$2" = '--list-samples' ]; then
          case "$1" in
            *bad-discovery.wiff) echo 'damaged sample index'; exit 1 ;;
            *multi.wiff) echo '{"samples":[{"id":7},{"id":2}]}' ;;
            *) echo '{"samples":[{"id":1}]}' ;;
          esac
          exit 0
        fi
        """#
        try Data(("#!/bin/sh\n" + discovery + "\n" + body + "\n").utf8).write(to: path)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: path.path)
        return path
    }

    private func waitUntil(_ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(8)
        while !predicate(), Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(predicate(), "Timed out waiting for converter state")
    }

    private func started() throws -> [String] {
        try String(contentsOf: folder.appendingPathComponent("started.txt"), encoding: .utf8)
            .split(separator: "\n").map(String.init)
    }

    func testMultiSampleSelectionUsesMetadataIDsAndFinalNames() async throws {
        let model = ConversionModel(executable: try converter(#"""
            printf '%s\n' "$*" >> started.txt
            test "$6" = '--sample' || exit 8
            target="${3%.sp.mzML}_$7.sp.mzML"
            set -C
            echo selected > "$target"
            """#))
        model.select(input("multi.wiff"))
        model.profile = .sparse
        model.start()
        try await waitUntil { model.sampleSelection != nil }
        XCTAssertEqual(model.sampleSelection?.files.first?.samples, [2, 7])
        XCTAssertFalse(FileManager.default.fileExists(atPath: input("started.txt").path))
        XCTAssertEqual(model.sampleDestination(for: model.input!, id: 7)?.lastPathComponent, "multi_7.sp.mzML")
        model.selectSamples([model.input!: []])
        XCTAssertNotNil(model.sampleSelection, "An empty selection must not start conversion")
        model.selectSamples([model.input!: [99]])
        XCTAssertNotNil(model.sampleSelection, "A stale/unknown ID must not start conversion")
        model.selectSamples([model.input!: [7]])
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.sampleID), [7])
        XCTAssertEqual(model.results.map(\.outcome), [.converted])
        XCTAssertEqual(model.convertedOutputs.map(\.lastPathComponent), ["multi_7.sp.mzML"])
        XCTAssertTrue(try started().first!.hasSuffix("--sample 7"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: input("multi_2.sp.mzML").path))
    }

    func testMultiSampleCancelBeforeConversionAndMixedQueueSelection() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("multi.wiff"), input("single.wiff"), input("bad-discovery.wiff")])
        model.start()
        try await waitUntil { model.sampleSelection != nil }
        XCTAssertTrue(model.sampleSelection!.includesSingleSampleFiles)
        model.cancel()
        XCTAssertFalse(model.running)
        XCTAssertNil(model.sampleSelection)
        XCTAssertFalse(FileManager.default.fileExists(atPath: input("started.txt").path))
        model.start()
        try await waitUntil { model.sampleSelection != nil }
        model.selectSamples([input("multi.wiff"): []])
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.skipped, .converted, .failed])
        XCTAssertEqual(try started().count, 1)
    }

    func testMultiSampleOverwriteCheckUsesSuffixedDestination() async throws {
        let model = ConversionModel(executable: try converter(#"""
            printf '%s\n' "$*" >> started.txt
            case "$7" in
              2) exit 1 ;;
              7) echo selected > "${3%.mzML}_$7.mzML" ;;
            esac
            """#))
        model.select(input("multi.wiff"))
        model.output = input("custom.mc.mzML")
        try Data("preserve".utf8).write(to: input("custom.mc_2.mzML"))
        model.start()
        try await waitUntil { model.sampleSelection != nil }
        model.selectSamples([model.input!: [2, 7]])
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.skipped, .converted])
        XCTAssertEqual(try String(contentsOf: input("custom.mc_2.mzML"), encoding: .utf8), "preserve")
        XCTAssertEqual(try started().count, 1)
        XCTAssertEqual(model.convertedOutputs.map(\.lastPathComponent), ["custom.mc_7.mzML"])
    }

    func testMultiSampleInspectionReportsEachSelectedRun() async throws {
        let model = ConversionModel(executable: try converter())
        model.select(input("multi.wiff"))
        model.inspect = true
        model.start()
        try await waitUntil { model.sampleSelection != nil }
        model.selectSamples([model.input!: [2, 7]])
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.sampleID), [2, 7])
        XCTAssertEqual(model.results.map(\.outcome), [.inspected, .inspected])
        XCTAssertEqual(model.status, "Batch complete: 2 inspected.")
        XCTAssertTrue(model.convertedOutputs.isEmpty)
        XCTAssertEqual(try started().count, 2)
        XCTAssertTrue(try started()[0].contains("|--inspect|--sample|2|"))
        XCTAssertTrue(try started()[1].contains("|--inspect|--sample|7|"))
    }

    func testAgilentFoldersAreQueuedButOtherFoldersAndDotDFilesAreRejected() throws {
        let bundle = folder.appendingPathComponent("Run 1.d", isDirectory: true)
        let upper = folder.appendingPathComponent("Run2.D", isDirectory: true)
        let plain = folder.appendingPathComponent("results", isDirectory: true)
        for directory in [bundle, upper, plain] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        FileManager.default.createFile(atPath: input("notes.d").path, contents: Data())
        let model = ConversionModel()
        model.addInputs([bundle, input("one.wiff"), upper, plain, input("notes.d")])
        XCTAssertEqual(model.inputs.map(\.lastPathComponent), ["Run 1.d", "one.wiff", "Run2.D"])
        XCTAssertTrue(model.status.contains("2 rejected"))
        XCTAssertEqual(model.output?.lastPathComponent, "Run 1.mzML")
        XCTAssertEqual(model.output?.hasDirectoryPath, false)
        XCTAssertEqual(model.destination(for: upper)?.lastPathComponent, "Run2.mzML")
    }

    func testMultipleSelectionDeduplicationFilteringAndRemoval() {
        let model = ConversionModel()
        model.select(input("single.wiff"))
        model.inspect = true
        model.addInputs([input("single.wiff"), input("second.WIFF2"), input("third.wiff"), input("sidecar.scan")])
        XCTAssertEqual(model.inputs.map(\.lastPathComponent), ["single.wiff", "second.WIFF2", "third.wiff"])
        XCTAssertTrue(model.isBatch)
        XCTAssertFalse(model.inspect)
        XCTAssertTrue(model.status.contains("1 rejected"))
        model.removeInputs([input("single.wiff"), input("third.wiff")])
        XCTAssertEqual(model.inputs, [input("second.WIFF2")])
        XCTAssertFalse(model.isBatch)
        XCTAssertEqual(model.output?.lastPathComponent, "second.mzML")
        model.removeInputs([input("second.WIFF2")])
        XCTAssertNil(model.input)
        XCTAssertNil(model.outputDirectory)
        model.select(input("invalid.txt"))
        XCTAssertTrue(model.inputs.isEmpty)
    }

    func testFinderMultipleFileDeliveryBeforeAndAfterModelAttachment() {
        let delegate = ConvertDelegate()
        let model = ConversionModel()
        delegate.application(NSApplication.shared, open: [input("one.wiff"), input("two.wiff2")])
        delegate.application(NSApplication.shared, open: [input("three.wiff")])
        delegate.attach(model)
        XCTAssertEqual(model.inputs.map(\.lastPathComponent), ["one.wiff", "two.wiff2", "three.wiff"])
        delegate.application(NSApplication.shared, open: [input("two.wiff2"), input("four.wiff")])
        XCTAssertEqual(model.inputs.map(\.lastPathComponent), ["one.wiff", "two.wiff2", "three.wiff", "four.wiff"])
    }

    func testSharedDirectoryAndModeNamesRetainSourceStem() {
        let model = ConversionModel()
        model.addInputs([input("one.wiff"), input("two.wiff2"), input("three.sp.wiff")])
        let destination = folder.appendingPathComponent("outputs", isDirectory: true)
        model.setOutputDirectory(destination)
        XCTAssertEqual(model.destination(for: model.inputs[1]), destination.appendingPathComponent("two.mzML"))
        model.profile = .sparse
        XCTAssertEqual(model.inputs.map { model.destination(for: $0)?.lastPathComponent }, ["one.sp.mzML", "two.sp.mzML", "three.sp.sp.mzML"])
        XCTAssertTrue(model.inputs.allSatisfy { model.destination(for: $0)?.deletingLastPathComponent() == destination })
        model.profile = .full
        XCTAssertEqual(model.destination(for: model.inputs[2])?.lastPathComponent, "three.sp.mzML")
    }

    func testSequentialBatchUsesOneModeAndReportsProgressAndSummary() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("one.wiff"), input("two.wiff2")])
        model.profile = .sparse
        var exit: Int32?
        model.onCompletion = { exit = $0 }
        model.start()
        try await waitUntil { model.progress != nil }
        XCTAssertGreaterThan(model.overallProgress, 0)
        model.addInputs([input("ignored.wiff")])
        model.removeInputs([input("two.wiff2")])
        model.setOutputDirectory(input("ignored"))
        XCTAssertEqual(model.inputs.count, 2)
        XCTAssertEqual(model.outputDirectory, folder)
        try await waitUntil { !model.running }
        XCTAssertEqual(exit, 0)
        XCTAssertEqual(model.results.map(\.outcome), [.converted, .converted])
        XCTAssertEqual(model.completedCount, 2)
        XCTAssertEqual(model.overallProgress, 2)
        XCTAssertEqual(model.status, "Batch complete: 2 converted.")
        XCTAssertEqual(try started().count, 2)
        XCTAssertTrue(try started().allSatisfy { $0.hasSuffix("|--profile|sparse") })
        XCTAssertEqual(model.convertedOutputs.map(\.lastPathComponent), ["one.sp.mzML", "two.sp.mzML"])
        XCTAssertTrue(model.log.contains("[1/2] one.wiff"))
        XCTAssertTrue(model.log.contains("[2/2] two.wiff2"))
        XCTAssertFalse(model.log.contains("mkdir:"), "Two processes must never run at once")
    }

    func testExistingFilesDirectoriesAndBrokenSymlinksAreSkipped() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("one.wiff"), input("two.wiff"), input("three.wiff"), input("four.wiff")])
        let original = Data("preserve".utf8)
        try original.write(to: input("one.mzML"))
        try FileManager.default.createSymbolicLink(at: input("two.mzML"), withDestinationURL: input("absent"))
        try FileManager.default.createDirectory(at: input("three.mzML"), withIntermediateDirectories: false)
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.skipped, .skipped, .skipped, .converted])
        XCTAssertEqual(try Data(contentsOf: input("one.mzML")), original)
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: input("two.mzML").path), input("absent").path)
        XCTAssertEqual(try started().count, 1)
        XCTAssertEqual(model.status, "Batch complete: 1 converted, 3 skipped.")
    }

    func testDuplicateBasenamesUseQueueOrderWithoutOverwrite() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("a/sample.wiff"), input("b/sample.wiff2")])
        model.setOutputDirectory(folder)
        model.profile = .sparse
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.converted, .skipped])
        XCTAssertEqual(try started().count, 1)
        XCTAssertEqual(try String(contentsOf: input("sample.sp.mzML"), encoding: .utf8), "fixture:sparse")
    }

    func testFileFailureDoesNotPreventLaterConversion() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("one.wiff"), input("fail.wiff"), input("three.wiff2")])
        var exit: Int32?
        model.onCompletion = { exit = $0 }
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.converted, .failed, .converted])
        XCTAssertEqual(exit, 1)
        XCTAssertFalse(model.succeeded)
        XCTAssertEqual(model.status, "Batch complete: 2 converted, 1 failed.")
        XCTAssertTrue(model.log.contains("fixture file failed"))
        XCTAssertEqual(try started().count, 3)
        XCTAssertEqual(model.convertedOutputs.count, 2)
    }

    func testProcessLaunchFailuresContinueAndCompleteOnce() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("one.wiff"), input("two.wiff")])
        model.setOutputDirectory(input("missing-directory"))
        var completions = 0
        model.onCompletion = { _ in completions += 1 }
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.failed, .failed])
        XCTAssertEqual(completions, 1)
        XCTAssertTrue(model.log.contains("Could not start conversion:"))
    }

    func testCancellationCleansActiveWorkAndDoesNotStartRemainingFiles() async throws {
        let executable = try converter(#"""
            printf '%s\n' "$1" >> started.txt
            trap 'rm -f "$3.partial"; echo "kmzconvert: cancelled"; exit 130' INT
            touch "$3.partial"
            printf '\rConverting 1/100 spectra'
            while :; do sleep 0.05; done
            """#)
        let model = ConversionModel(executable: executable)
        model.addInputs([input("one.wiff"), input("two.wiff2"), input("three.wiff")])
        var exit: Int32?
        model.onCompletion = { exit = $0 }
        model.start()
        try await waitUntil { model.progress != nil }
        model.cancel()
        model.cancel()
        try await waitUntil { !model.running }
        XCTAssertEqual(exit, 130)
        XCTAssertEqual(model.results.map(\.outcome), [.cancelled, .notRun, .notRun])
        XCTAssertEqual(try started().count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: input("one.mzML.partial").path))
        XCTAssertEqual(model.status, "Batch cancelled: 1 cancelled, 2 not run.")
        XCTAssertFalse(model.succeeded)
        XCTAssertFalse(model.cancelling)
    }

    func testCancellationDuringDiscoveryStopsQueue() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("one.wiff"), input("two.wiff")])
        try Data("existing".utf8).write(to: input("one.mzML"))
        model.start() // Discovery must finish before per-sample output checks.
        model.cancel()
        try await waitUntil { !model.running }
        XCTAssertFalse(model.running)
        XCTAssertEqual(model.results.map(\.outcome), [.cancelled, .notRun])
        XCTAssertFalse(FileManager.default.fileExists(atPath: input("started.txt").path))
    }

    func testNextFileDoesNotReusePreviousSpectrumProgress() async throws {
        let model = ConversionModel(executable: try converter(#"""
            case "$1" in
              *two.wiff) trap 'exit 130' INT; touch second-started; while :; do sleep 0.05; done ;;
            esac
            printf '\rConverting 4/4 spectra\n'
            echo fixture > "$3"
            """#))
        model.addInputs([input("one.wiff"), input("two.wiff")])
        model.start()
        try await waitUntil { FileManager.default.fileExists(atPath: self.input("second-started").path) }
        XCTAssertEqual(model.results.map(\.outcome), [.converted, .converting])
        XCTAssertNil(model.progress)
        XCTAssertEqual(model.overallProgress, 1)
        model.cancel()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.converted, .cancelled])
    }

    func testSingleFileCustomOutputModeAndInspectRemainAvailable() async throws {
        let model = ConversionModel(executable: try converter())
        model.select(input("one.wiff2"))
        model.output = input("custom.mzML")
        model.profile = .sparse
        XCTAssertEqual(model.output, input("custom.sp.mzML"))
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.status, "Conversion complete.")
        XCTAssertTrue(FileManager.default.fileExists(atPath: input("custom.sp.mzML").path))
        model.inspect = true
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.status, "Inspection complete.")
        XCTAssertEqual(model.results.map(\.outcome), [.inspected])
        XCTAssertTrue(try started().last!.hasSuffix("|--inspect|||"))
        XCTAssertTrue(model.log.contains("{\"spectra\":4}"))
        XCTAssertTrue(model.convertedOutputs.isEmpty)
    }

    func testSingleFileExistingOutputStillUsesBackendNoClobberError() async throws {
        let model = ConversionModel(executable: try converter())
        model.select(input("one.wiff"))
        let original = Data("keep".utf8)
        try original.write(to: input("one.mzML"))
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.failed])
        XCTAssertEqual(try Data(contentsOf: input("one.mzML")), original)
        XCTAssertEqual(try started().count, 1)
    }

    func testAllSkippedAndRerunHaveFreshState() async throws {
        let model = ConversionModel(executable: try converter())
        model.addInputs([input("one.wiff"), input("two.wiff")])
        model.start()
        try await waitUntil { !model.running }
        model.start()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.skipped, .skipped])
        XCTAssertEqual(model.status, "Batch complete: 2 skipped.")
        XCTAssertTrue(model.succeeded)
        XCTAssertFalse(model.log.contains("Wrote "))
        XCTAssertEqual(try started().count, 2)
    }

    func testBetaReferenceValidationAndInspection() throws {
        XCTAssertEqual(try MassReferenceEntry.parse(" 200;3\n\n250;0 ").count, 2)
        for text in ["", "200", "200;3;4", "nan;3", "200;inf", "0;3", "-2;3", "200;-1", "200;3\n2e2;3.0"] {
            XCTAssertThrowsError(try MassReferenceEntry.parse(text), text)
        }
        let model = ConversionModel()
        XCTAssertFalse(model.massCorrectionEnabled)
        model.massCorrectionEnabled = true
        XCTAssertNotNil(model.referenceValidationMessage)
        model.inspect = true
        XCTAssertNil(model.referenceValidationMessage)
    }

    func testBetaAutomaticNamesAndExplicitDestination() {
        let model = ConversionModel()
        model.select(input("one.wiff"))
        model.massCorrectionEnabled = true
        XCTAssertEqual(model.output?.lastPathComponent, "one.mc.mzML")
        model.profile = .sparse
        XCTAssertEqual(model.output?.lastPathComponent, "one.mc.sp.mzML")
        model.massCorrectionEnabled = false
        XCTAssertEqual(model.output?.lastPathComponent, "one.sp.mzML")
        model.output = input("custom.sp.mzML")
        model.massCorrectionEnabled = true
        XCTAssertEqual(model.output?.lastPathComponent, "custom.sp.mzML")
        model.addInputs([input("two.wiff")])
        XCTAssertEqual(model.destination(for: input("two.wiff"))?.lastPathComponent, "two.mc.sp.mzML")
    }

    func testBetaBatchSnapshotsRepeatedReferencesAndInspectOmitsThem() async throws {
        let model = ConversionModel(executable: try converter(#"""
            printf '%s\n' "$*" >> started.txt
            sleep 0.05
            if [ "$2" = '--inspect' ]; then echo '{"spectra":4}'; else echo fixture > "$3"; fi
            """#))
        model.addInputs([input("one.wiff"), input("two.wiff")])
        model.massCorrectionEnabled = true
        model.massReferenceText = "200;3\n250;4"
        model.start()
        model.massReferenceText = "bad"
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.converted, .converted])
        XCTAssertTrue(try started().allSatisfy { $0.hasSuffix("--mass-reference 200.0;3.0 --mass-reference 250.0;4.0") })
        XCTAssertEqual(model.convertedOutputs.map(\.lastPathComponent), ["one.mc.mzML", "two.mc.mzML"])
        model.removeInputs([input("two.wiff")])
        model.inspect = true
        model.start()
        try await waitUntil { !model.running }
        XCTAssertTrue(try started().last!.hasSuffix("--inspect"))
        XCTAssertFalse(try started().last!.contains("--mass-reference"))
    }

    func testBetaCancellationWhileEstimatingBeforeSpectrumProgress() async throws {
        let model = ConversionModel(executable: try converter(#"""
            printf '%s\n' "$1" >> started.txt
            trap 'echo "kmzconvert: cancelled"; exit 130' INT
            echo 'Estimating mass reference 1/1: 200;3'
            while :; do sleep 0.05; done
            """#))
        model.addInputs([input("one.wiff"), input("two.wiff")])
        model.massCorrectionEnabled = true
        model.massReferenceText = "200;3"
        model.start()
        try await waitUntil { model.log.contains("Estimating mass reference") }
        XCTAssertNil(model.progress)
        model.cancel()
        try await waitUntil { !model.running }
        XCTAssertEqual(model.results.map(\.outcome), [.cancelled, .notRun])
        XCTAssertEqual(try started().count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: input("one.mc.mzML").path))
    }

    func testBetaSparseRunRetainsAutomaticNamingAfterToggle() async throws {
        let model = ConversionModel(executable: try converter())
        model.select(input("one.wiff"))
        model.massCorrectionEnabled = true
        model.massReferenceText = "200;3"
        model.profile = .sparse
        model.start()
        try await waitUntil { !model.running }
        model.massCorrectionEnabled = false
        XCTAssertEqual(model.output?.lastPathComponent, "one.sp.mzML")
    }
}
