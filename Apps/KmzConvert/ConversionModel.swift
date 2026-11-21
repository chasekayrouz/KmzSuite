import AppKit
import Foundation
import KmzMacSupport
import UniformTypeIdentifiers

enum ConversionOutcome: String {
    case pending = "Queued", converting = "Converting", converted = "Converted"
    case skipped = "Skipped", failed = "Failed", cancelled = "Cancelled", notRun = "Not run", inspected = "Inspected"
}

struct ConversionResult {
    let input: URL
    let output: URL?
    var sampleID: Int? = nil
    var baseOutput: URL? = nil
    var outcome: ConversionOutcome = .pending
    var message = ""
    var label: String { input.lastPathComponent + (sampleID.map { " · Sample \($0)" } ?? "") }
}

struct WiffSample: Decodable, Hashable { let id: Int }
struct WiffSampleList: Decodable { let samples: [WiffSample] }
struct SampleSelectionFile: Identifiable {
    let input: URL
    let samples: [Int]
    var id: URL { input }
}
struct SampleSelectionRequest: Identifiable {
    let id = UUID()
    let files: [SampleSelectionFile]
    let includesSingleSampleFiles: Bool
}

struct MassReferenceEntry: Hashable {
    let mz: Double
    let rt: Double
    var argument: String { "\(mz);\(rt)" }

    static func parse(_ text: String) throws -> [MassReferenceEntry] {
        var entries: [MassReferenceEntry] = []
        var seen = Set<MassReferenceEntry>()
        for line in text.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            let parts = trimmed.components(separatedBy: ";").map { $0.trimmingCharacters(in: .whitespaces) }
            guard parts.count == 2, let mz = Double(parts[0]), let rt = Double(parts[1]),
                  mz.isFinite, mz > 0, rt.isFinite, rt >= 0 else {
                throw NSError(domain: "KmzConvert", code: 1, userInfo: [NSLocalizedDescriptionKey:
                    "Invalid reference ‘\(trimmed)’. Use finite positive expected_mz;nonnegative RT_minutes."])
            }
            let entry = MassReferenceEntry(mz: mz, rt: rt)
            guard seen.insert(entry).inserted else {
                throw NSError(domain: "KmzConvert", code: 1, userInfo: [NSLocalizedDescriptionKey: "Duplicate reference ‘\(trimmed)’."])
            }
            entries.append(entry)
        }
        guard !entries.isEmpty else {
            throw NSError(domain: "KmzConvert", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Enabled mass correction requires at least one expected_mz;RT_minutes reference."])
        }
        return entries
    }
}

// Read callbacks collect bytes before dispatching UI updates, so termination cannot
// lose queued messages or attribute the previous file's progress to the next one.
private final class ProcessLog {
    private let lock = NSLock()
    private var data = Data()
    func read(_ handle: FileHandle, toEnd: Bool = false) {
        lock.lock(); defer { lock.unlock() }
        let bytes = toEnd ? (try? handle.readToEnd()) : handle.availableData
        if let bytes { data.append(bytes) }
    }
    func snapshot() -> Data { lock.lock(); defer { lock.unlock() }; return data }
}

@MainActor final class ConversionModel: ObservableObject {
    @Published private(set) var inputs: [URL] = []
    var input: URL? { inputs.first }
    var isBatch: Bool { inputs.count > 1 }
    @Published var output: URL? { didSet { outputIsCustom = true } }
    @Published private(set) var outputDirectory: URL?
    @Published var profile: OutputProfile = .full {
        didSet {
            let custom = outputIsCustom
            if let output { self.output = profile.outputURL(output); outputIsCustom = custom }
            resetResults()
        }
    }
    static let massCorrectionWarning = "Beta mass correction: applies one global offset to measured MS1 masses. Accuracy across the run is not guaranteed. MS2 masses remain uncorrected."
    @Published var massCorrectionEnabled = false {
        didSet { refreshCorrectionName(); resetResults() }
    }
    @Published var massReferenceText = "" { didSet { resetResults() } }
    @Published var inspect = false
    @Published private(set) var running = false
    @Published private(set) var cancelling = false
    @Published private(set) var progress: ConversionProgress?
    @Published private(set) var results: [ConversionResult] = []
    @Published private(set) var status = "Choose SCIEX WIFF files or Agilent .d folders."
    @Published private(set) var log = ""
    @Published private(set) var succeeded = false
    @Published private(set) var sampleSelection: SampleSelectionRequest?
    private let executable: URL?
    private var process: Process?
    private var pipe: Pipe?
    private var processLog: ProcessLog?
    private var logData = Data()
    private var nextIndex = 0
    private var activeIndex: Int?
    private var runProfile: OutputProfile = .full
    private var runInspect = false
    private var runIsBatch = false
    private var runMassReferences: [String] = []
    private var discovering = false
    private var discoveredSamples: [URL: [Int]] = [:]
    private var outputIsCustom = false
    var onCompletion: ((Int32) -> Void)?

    init(executable: URL? = NativeSupport.converter) { self.executable = executable }

    var completedCount: Int { results.filter { ![.pending, .converting, .notRun].contains($0.outcome) }.count }
    var overallProgress: Double {
        Double(completedCount) + (progress.map { Double($0.done) / Double($0.total) } ?? 0)
    }
    var convertedOutputs: [URL] { results.filter { $0.outcome == .converted }.compactMap(\.output) }
    var referenceValidationMessage: String? {
        guard massCorrectionEnabled, !inspect else { return nil }
        do { _ = try MassReferenceEntry.parse(massReferenceText); return nil }
        catch { return error.localizedDescription }
    }

    private func refreshCorrectionName() {
        guard !running, !outputIsCustom, let input, let outputDirectory else { return }
        output = outputDirectory.appendingPathComponent(profile.suggestedOutputURL(for: input, massCorrection: massCorrectionEnabled).lastPathComponent)
        outputIsCustom = false
    }

    func destination(for input: URL) -> URL? {
        if !isBatch { return output }
        return outputDirectory?.appendingPathComponent(profile.suggestedOutputURL(for: input, massCorrection: massCorrectionEnabled).lastPathComponent)
    }

    func select(_ url: URL) {
        guard !running else { return }
        guard accepts(url) else { status = "Select a .wiff or .wiff2 file, or an Agilent .d folder."; return }
        inputs = []
        outputDirectory = nil
        addInputs([url])
    }

    func addInputs(_ urls: [URL]) {
        guard !running, !urls.isEmpty else { return }
        var rejected = 0
        for url in urls {
            guard accepts(url) else { rejected += 1; continue }
            let normalized = url.standardizedFileURL
            guard !inputs.contains(normalized) else { continue }
            if inputs.isEmpty {
                outputDirectory = normalized.deletingLastPathComponent()
                output = profile.suggestedOutputURL(for: normalized, massCorrection: massCorrectionEnabled)
                outputIsCustom = false
            }
            inputs.append(normalized)
        }
        if isBatch { inspect = false }
        resetResults()
        status = rejected > 0 ? "Only .wiff and .wiff2 files and Agilent .d folders can be added (\(rejected) rejected)." :
            "\(inputs.count) input(s) ready. WIFF sidecars must be beside each WIFF file."
    }

    private func accepts(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
        let ext = url.pathExtension.lowercased()
        // An Agilent MassHunter acquisition is a .d folder; the converter checks its contents.
        return isDirectory ? ext == "d" : ["wiff", "wiff2"].contains(ext)
    }

    func removeInputs(_ urls: Set<URL>) {
        guard !running else { return }
        inputs.removeAll { urls.contains($0) }
        if let input {
            output = outputDirectory?.appendingPathComponent(profile.suggestedOutputURL(for: input, massCorrection: massCorrectionEnabled).lastPathComponent)
        } else { output = nil; outputDirectory = nil }
        outputIsCustom = false
        resetResults()
        status = inputs.isEmpty ? "Choose SCIEX WIFF files or Agilent .d folders." : "\(inputs.count) input(s) ready."
    }

    private func resetResults() {
        guard !running else { return }
        results = []; progress = nil; succeeded = false
    }

    func chooseInput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = ["wiff", "wiff2"].compactMap { UTType(filenameExtension: $0) }
        panel.message = "Add .wiff or .wiff2 entry points, or Agilent .d folders, to the queue"
        if panel.runModal() == .OK { addInputs(panel.urls) }
    }

    func setOutputDirectory(_ directory: URL) {
        guard !running else { return }
        outputDirectory = directory
        if let input { output = directory.appendingPathComponent(profile.suggestedOutputURL(for: input, massCorrection: massCorrectionEnabled).lastPathComponent) }
        outputIsCustom = false
        resetResults()
    }

    func chooseOutputDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.directoryURL = outputDirectory
        panel.message = "Choose one output folder for all queued files"
        if panel.runModal() == .OK, let url = panel.url { setOutputDirectory(url) }
    }

    func chooseOutput() {
        guard !running, !isBatch else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = output?.lastPathComponent ?? "converted.mzML"
        panel.directoryURL = outputDirectory
        panel.message = "Choose a new \(profile == .full ? ".mzML" : ".sp.mzML") file. Existing files are never overwritten."
        if panel.runModal() == .OK, let url = panel.url {
            output = profile.outputURL(url)
            outputDirectory = output?.deletingLastPathComponent()
            resetResults()
        }
    }

    func start() {
        guard !running, !inputs.isEmpty, !inspect || !isBatch,
              inspect || (isBatch ? outputDirectory != nil : output != nil) else { return }
        if let error = referenceValidationMessage { status = error; return }
        runMassReferences = (!inspect && massCorrectionEnabled) ? (try? MassReferenceEntry.parse(massReferenceText).map(\.argument)) ?? [] : []
        if profile == .sparse, let output {
            let custom = outputIsCustom
            self.output = profile.outputURL(output)
            outputIsCustom = custom
        }
        guard let executable, FileManager.default.isExecutableFile(atPath: executable.path) else {
            status = "Bundled converter is unavailable. Rebuild KmzConvert.app."; return
        }
        results = inputs.map { ConversionResult(input: $0, output: inspect ? nil : destination(for: $0)) }
        runProfile = profile; runInspect = inspect; runIsBatch = isBatch
        progress = nil; logData = Data(); log = ""; succeeded = false; cancelling = false
        nextIndex = 0; activeIndex = nil; running = true
        discovering = true; discoveredSamples = [:]; sampleSelection = nil
        startNext()
    }

    private func scheduleNext() {
        Task { @MainActor [weak self] in
            await Task.yield()
            guard let self, self.running, self.process == nil else { return }
            self.startNext()
        }
    }

    private func startNext() {
        guard running else { return }
        if cancelling { completeRun(); return }
        if nextIndex == results.count {
            if discovering { finishDiscovery() } else { completeRun() }
            return
        }
        let index = nextIndex
        nextIndex += 1
        let result = results[index]
        if result.outcome == .failed || result.outcome == .skipped { scheduleNext(); return }
        if !discovering, runIsBatch, let output = result.output,
           (try? FileManager.default.attributesOfItem(atPath: output.path)) != nil {
            results[index].outcome = .skipped
            results[index].message = "Output already exists (including symlinks)."
            appendLog("[\(index + 1)/\(results.count)] \(result.input.lastPathComponent): skipped — \(output.lastPathComponent) already exists.\n")
            scheduleNext()
            return
        }
        let job = Process(), stream = Pipe(), buffer = ProcessLog()
        job.executableURL = executable
        job.arguments = [result.input.path] + (discovering ? ["--list-samples"] : runInspect ? ["--inspect"] :
            ["--output", (result.baseOutput ?? result.output!).path, "--profile", runProfile.rawValue])
        if !discovering {
            if let sampleID = result.sampleID { job.arguments! += ["--sample", String(sampleID)] }
            if !runInspect { job.arguments! += runMassReferences.flatMap { ["--mass-reference", $0] } }
        }
        job.currentDirectoryURL = result.output?.deletingLastPathComponent() ?? result.input.deletingLastPathComponent()
        var environment = ProcessInfo.processInfo.environment
        for key in environment.keys where key.hasPrefix("PYTHON") || key.hasPrefix("DYLD_") || key.hasPrefix("_PYI") { environment.removeValue(forKey: key) }
        environment["PYTHONUNBUFFERED"] = "1"
        job.environment = environment
        job.standardOutput = stream; job.standardError = stream
        process = job; pipe = stream; processLog = buffer; activeIndex = index
        progress = nil
        results[index].outcome = .converting
        status = discovering ? "Reading samples · \(result.input.lastPathComponent)…" :
            "Run \(index + 1)/\(results.count): \(result.label) — \(runInspect ? "inspecting metadata" : "reading acquisition")…"
        if !discovering {
            appendLog("[\(index + 1)/\(results.count)] \(result.label)\(result.output.map { " → " + $0.lastPathComponent } ?? " — inspect")\n")
        }
        stream.fileHandleForReading.readabilityHandler = { [weak self, weak job] handle in
            buffer.read(handle)
            Task { @MainActor in
                guard let self, let job, self.process === job else { return }
                self.refreshLog()
            }
        }
        job.terminationHandler = { [weak self] job in
            Task { @MainActor in self?.finish(job) }
        }
        do { try job.run() } catch {
            stream.fileHandleForReading.readabilityHandler = nil
            results[index].outcome = .failed
            results[index].message = "Could not start conversion: " + error.localizedDescription
            appendLog(results[index].message + "\n")
            process = nil; pipe = nil; processLog = nil; activeIndex = nil
            scheduleNext()
        }
    }

    private func appendLog(_ text: String) {
        logData.append(contentsOf: text.utf8)
        refreshLog()
    }

    private func refreshLog() {
        var displayed = logData
        if let processLog { displayed.append(processLog.snapshot()) }
        // Bound display memory without dropping the complete saved batch log.
        log = String(decoding: displayed.suffix(200_000), as: UTF8.self).replacingOccurrences(of: "\r", with: "\n")
        if !discovering, let processLog, let current = ConversionProgress.parse(String(decoding: processLog.snapshot().suffix(200_000), as: UTF8.self)) {
            progress = current
            if !cancelling, let index = activeIndex {
                status = "Run \(index + 1)/\(results.count): \(results[index].label) — converting \(current.done)/\(current.total) spectra…"
            }
        }
    }

    private func finish(_ job: Process) {
        guard running, process === job, let index = activeIndex else { return }
        pipe?.fileHandleForReading.readabilityHandler = nil
        if let handle = pipe?.fileHandleForReading { processLog?.read(handle, toEnd: true) }
        let data = processLog?.snapshot() ?? Data()
        if !discovering { logData.append(data) }
        let code = job.terminationStatus
        if discovering {
            if code == 0, let decoded = try? JSONDecoder().decode(WiffSampleList.self, from: data),
               !decoded.samples.isEmpty, decoded.samples.allSatisfy({ $0.id > 0 }),
               Set(decoded.samples.map(\.id)).count == decoded.samples.count {
                discoveredSamples[results[index].input] = decoded.samples.map(\.id).sorted()
                results[index].outcome = .pending
            } else {
                results[index].outcome = cancelling || code == 130 ? .cancelled : .failed
                results[index].message = "Could not read the input's samples. See the log."
                logData.append(data)
                appendLog("\n\(results[index].input.lastPathComponent): \(results[index].message)\n")
            }
            if code == 130 { cancelling = true }
            process = nil; pipe = nil; processLog = nil; activeIndex = nil; progress = nil
            refreshLog(); scheduleNext()
            return
        }
        results[index].outcome = code == 0 ? (runInspect ? .inspected : .converted) : (cancelling || code == 130 ? .cancelled : .failed)
        results[index].message = code == 0 ? (runInspect ? "Inspection complete." : "Conversion complete.") :
            (results[index].outcome == .cancelled ? "Cancelled; check the log for publication/recovery details." : "Conversion failed (exit \(code)). See the log.")
        if code == 130 { cancelling = true }
        process = nil; pipe = nil; processLog = nil; activeIndex = nil; progress = nil
        appendLog("\n\(results[index].label): \(results[index].message)\n")
        scheduleNext()
    }

    private func finishDiscovery() {
        discovering = false
        let files = results.compactMap { result -> SampleSelectionFile? in
            guard let ids = discoveredSamples[result.input], ids.count > 1 else { return nil }
            return SampleSelectionFile(input: result.input, samples: ids)
        }
        if files.isEmpty { selectSamples([:]); return }
        sampleSelection = SampleSelectionRequest(files: files,
            includesSingleSampleFiles: discoveredSamples.values.contains { $0.count == 1 })
        status = "Choose the samples/runs to \(runInspect ? "inspect" : "convert")."
    }

    func sampleDestination(for input: URL, id: Int) -> URL? {
        destination(for: input).map { runProfile.sampleOutputURL($0, sampleID: id) }
    }

    func selectSamples(_ selection: [URL: Set<Int>]) {
        guard running, !discovering, process == nil, !cancelling else { return }
        for (input, ids) in discoveredSamples where ids.count > 1 {
            guard let selected = selection[input], selected.isSubset(of: Set(ids)) else { return }
        }
        guard discoveredSamples.contains(where: { input, ids in
            ids.count == 1 || !(selection[input] ?? []).isEmpty
        }) || sampleSelection == nil else { return }
        results = results.flatMap { result -> [ConversionResult] in
            guard let ids = discoveredSamples[result.input] else { return [result] }
            guard ids.count > 1 else { return [result] }
            let selected = ids.filter { (selection[result.input] ?? []).contains($0) }
            if selected.isEmpty {
                var skipped = result
                skipped.outcome = .skipped; skipped.message = "No samples selected."
                return [skipped]
            }
            return selected.map { id in
                ConversionResult(input: result.input,
                    output: result.output.map { runProfile.sampleOutputURL($0, sampleID: id) },
                    sampleID: id, baseOutput: result.output)
            }
        }
        sampleSelection = nil
        runIsBatch = runIsBatch || results.count > 1 || results.contains { $0.sampleID != nil }
        nextIndex = 0
        startNext()
    }

    private func completeRun() {
        let wasCancelled = cancelling
        if wasCancelled {
            for index in results.indices where results[index].outcome == .pending { results[index].outcome = .notRun }
        }
        let failed = results.filter { $0.outcome == .failed }.count
        succeeded = !wasCancelled && failed == 0
        if runIsBatch {
            let counts = [(.converted, "converted"), (.inspected, "inspected"), (.skipped, "skipped"), (.failed, "failed"), (.cancelled, "cancelled"), (.notRun, "not run")] as [(ConversionOutcome, String)]
            let summary = counts.compactMap { outcome, title -> String? in
                let count = results.filter { $0.outcome == outcome }.count
                return count > 0 ? "\(count) \(title)" : nil
            }.joined(separator: ", ")
            status = "Batch \(wasCancelled ? "cancelled" : "complete"): \(summary)."
        } else if wasCancelled {
            status = "Cancelled. Check the log if output publication had already begun."
        } else { status = results.first?.message ?? "Complete." }
        appendLog(status + "\n")
        running = false; cancelling = false
        sampleSelection = nil; discovering = false
        onCompletion?(wasCancelled ? 130 : (failed > 0 ? 1 : 0))
    }

    func cancel() {
        guard running, !cancelling else { return }
        cancelling = true
        status = "Cancelling… waiting for the converter to clean up."
        if let process { process.interrupt() } else { completeRun() }
    }

    func saveLog() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "kmzconvert.log"
        if panel.runModal() == .OK, let url = panel.url {
            do { try logData.write(to: url) } catch { status = "Could not save log: " + error.localizedDescription }
        }
    }
}
