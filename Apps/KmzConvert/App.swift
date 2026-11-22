import AppKit
import SwiftUI
import UniformTypeIdentifiers
import KmzMacSupport

@MainActor final class ConvertDelegate: NSObject, NSApplicationDelegate {
    weak var model: ConversionModel?
    private var pending: [URL] = []
    func attach(_ model: ConversionModel) {
        self.model = model
        model.addInputs(pending)
        pending.removeAll()
    }
    func application(_ application: NSApplication, open urls: [URL]) {
        if let model { model.addInputs(urls) } else { pending.append(contentsOf: urls) }
    }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model, model.running else { return .terminateNow }
        model.onCompletion = { _ in sender.reply(toApplicationShouldTerminate: true) }
        model.cancel()
        return .terminateLater
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main struct KmzConvertApp: App {
    @NSApplicationDelegateAdaptor(ConvertDelegate.self) private var delegate
    @StateObject private var model = ConversionModel()
    var body: some Scene {
        WindowGroup {
            ConvertView(model: model).onAppear { delegate.attach(model) }
                .suiteTheme()
                .frame(minWidth: 720, minHeight: 780)
        }
        .defaultSize(width: 820, height: 860)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Add Acquisitions…") { model.chooseInput() }.keyboardShortcut("o").disabled(model.running)
            }
            CommandGroup(replacing: .help) {
                Button("KmzConvert Manual") { NativeSupport.showManual("KmzConvert") }
            }
        }
    }
}

struct ConvertView: View {
    @ObservedObject var model: ConversionModel
    @State private var targeted = false
    @State private var smokeStarted = false
    @State private var selectedInputs: Set<URL> = []
    private func displayPath(_ url: URL?) -> String {
        url.map { ($0.path as NSString).abbreviatingWithTildeInPath } ?? "Choose an input first"
    }
    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 12) {
            Text("KmzConvert").font(.largeTitle.bold())
            Text("SCIEX X500 WIFF and Agilent .d → indexed mzML").foregroundStyle(SuiteTheme.muted)
            GroupBox {
                HStack {
                    Text("Input queue · \(model.inputs.count) input(s)").font(.headline)
                    Spacer()
                    Button("Add Files…") { model.chooseInput() }.disabled(model.running)
                    Button("Remove") {
                        model.removeInputs(selectedInputs)
                        selectedInputs.removeAll()
                    }.disabled(model.running || selectedInputs.isEmpty)
                }
                if model.inputs.isEmpty {
                    Text("Drop WIFF or WIFF2 files, or Agilent .d folders, here. WIFF sidecars must be beside each WIFF file.")
                        .foregroundStyle(SuiteTheme.muted).frame(maxWidth: .infinity, minHeight: 110)
                } else {
                    List(selection: $selectedInputs) {
                        ForEach(model.inputs, id: \.self) { input in
                            let results = model.results.filter { $0.input == input }
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(input.lastPathComponent).fontWeight(.medium)
                                    Text(displayPath(input.deletingLastPathComponent())).font(.caption)
                                        .foregroundStyle(SuiteTheme.muted).lineLimit(1).truncationMode(.middle)
                                    if results.contains(where: { $0.sampleID != nil }) {
                                        ForEach(Array(results.enumerated()), id: \.offset) { _, result in
                                            Text("Sample \(result.sampleID.map(String.init) ?? "–") · \(result.outcome.rawValue)" +
                                                 (result.output.map { " → " + $0.lastPathComponent } ?? ""))
                                                .font(.caption).foregroundStyle(SuiteTheme.muted)
                                        }
                                    } else if let output = model.destination(for: input) {
                                        Text("→ " + output.lastPathComponent).font(.caption).foregroundStyle(SuiteTheme.muted)
                                    }
                                }
                                Spacer()
                                if results.count <= 1 { Text(results.first?.outcome.rawValue ?? "Queued").font(.caption) }
                            }
                            .help(results.isEmpty ? input.path : results.map { $0.label + ": " + $0.message }.joined(separator: "\n"))
                            .tag(input)
                        }
                    }
                    .listStyle(.plain).scrollContentBackground(.hidden).frame(height: 130)
                }
            }
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(targeted ? Color.accentColor : .clear, lineWidth: 2))
            .onDrop(of: [UTType.fileURL.identifier], isTargeted: $targeted) { providers in
                guard !model.running, !providers.isEmpty else { return false }
                Task { @MainActor in
                    var urls: [URL] = []
                    for provider in providers {
                        let url: URL? = await withCheckedContinuation { continuation in
                            _ = provider.loadObject(ofClass: URL.self) { url, _ in continuation.resume(returning: url) }
                        }
                        if let url { urls.append(url) }
                    }
                    model.addInputs(urls)
                }
                return true
            }
            Picker("Output", selection: $model.profile) {
                ForEach(OutputProfile.allCases) { profile in Text(profile.title).tag(profile) }
            }
            .pickerStyle(.radioGroup).disabled(model.running || model.inspect)
            Text(model.profile.detail).font(.caption).foregroundStyle(SuiteTheme.muted)
            GroupBox {
                VStack(alignment: .leading, spacing: 6) {
                    Toggle("Mass correction (beta)", isOn: $model.massCorrectionEnabled)
                        .disabled(model.running || model.inspect)
                    if model.massCorrectionEnabled {
                        Text("References · expected_mz;RT_minutes · one pair per line")
                            .font(.caption).foregroundStyle(SuiteTheme.muted)
                        TextEditor(text: $model.massReferenceText)
                            .font(.system(.caption, design: .monospaced)).frame(height: 65)
                            .disabled(model.running || model.inspect)
                            .accessibilityLabel("Mass correction references")
                        Text(ConversionModel.massCorrectionWarning).font(.caption).foregroundStyle(.orange)
                        if let error = model.referenceValidationMessage {
                            Text(error).font(.caption).foregroundStyle(.red)
                        }
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(model.isBatch ? "Output folder · all files" : "Destination").font(.headline)
                    Text(displayPath(model.isBatch ? model.outputDirectory : model.output)).font(.caption)
                        .foregroundStyle(SuiteTheme.muted).textSelection(.enabled)
                }
                Spacer()
                Button("Choose Folder…") { model.chooseOutputDirectory() }.disabled(model.running || model.inspect || model.input == nil)
                if !model.isBatch {
                    Button("Choose Output…") { model.chooseOutput() }.disabled(model.running || model.inspect || model.input == nil)
                }
            }
            Toggle("Inspect metadata without converting (single file)", isOn: $model.inspect).disabled(model.running || model.isBatch)
            Text("Experimental: qualified X500 positive-mode profile / dependent MS/MS data. Agilent folders saved without profile data are written as their stored centroids. Existing output files are never overwritten.")
                .font(.caption).foregroundStyle(SuiteTheme.muted)
            if model.running {
                if model.results.count > 1 {
                    ProgressView("Overall · \(model.completedCount)/\(model.results.count) jobs finished", value: model.overallProgress, total: Double(model.results.count))
                }
                if let progress = model.progress {
                    ProgressView("Current file · \(progress.done)/\(progress.total) spectra", value: Double(progress.done), total: Double(progress.total))
                } else { ProgressView().controlSize(.small) }
            }
            Text(model.status).textSelection(.enabled)
            HStack {
                Button(model.inspect ? "Inspect" : model.isBatch ? "Convert Batch" : "Convert") { model.start() }
                    .buttonStyle(SuiteButtonStyle(prominent: true)).disabled(model.running || model.input == nil || model.referenceValidationMessage != nil)
                Button("Cancel") { model.cancel() }.disabled(!model.running || model.cancelling)
                if !model.running && !model.inspect && !model.convertedOutputs.isEmpty {
                    Button(model.convertedOutputs.count > 1 ? "Reveal Outputs" : "Reveal Output") { NSWorkspace.shared.activateFileViewerSelecting(model.convertedOutputs) }
                }
                Spacer()
                Button("Save Log…") { model.saveLog() }.disabled(model.log.isEmpty || model.running)
            }
            ScrollView {
                Text(model.log.isEmpty ? "Conversion messages and inspection results appear here." : model.log)
                    .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 150).padding(10)
            .background(SuiteTheme.paper, in: RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(SuiteTheme.border, lineWidth: 1))
        }
        .padding(24)
        }
        .sheet(item: Binding(get: { model.sampleSelection }, set: { value in
            if value == nil && model.sampleSelection != nil { model.cancel() }
        })) { request in
            SampleSelectionView(model: model, request: request)
        }
        .task {
            guard !smokeStarted, let folder = NativeSupport.argument("--smoke-output") else { return }
            smokeStarted = true
            let directory = URL(fileURLWithPath: folder, isDirectory: true)
            try? await Task.sleep(for: .milliseconds(700))
            let arguments = CommandLine.arguments
            let inputs = arguments.indices.compactMap { index -> URL? in
                guard arguments[index] == "--smoke-input", arguments.indices.contains(index + 1) else { return nil }
                return URL(fileURLWithPath: arguments[index + 1])
            }
            if !inputs.isEmpty {
                model.addInputs(inputs)
                if NativeSupport.argument("--smoke-profile") == "sparse" { model.profile = .sparse }
                let references = arguments.indices.compactMap { index -> String? in
                    guard arguments[index] == "--smoke-mass-reference", arguments.indices.contains(index + 1) else { return nil }
                    return arguments[index + 1]
                }
                if !references.isEmpty {
                    model.massReferenceText = references.joined(separator: "\n")
                    model.massCorrectionEnabled = true
                }
                model.inspect = CommandLine.arguments.contains("--smoke-inspect")
                model.setOutputDirectory(directory)
                if !model.isBatch { model.output = model.profile.outputURL(directory.appendingPathComponent("converted.mzML")) }
                model.onCompletion = { code in
                    try? NativeSupport.report(["ok": code == 0, "exit": code, "status": model.status, "log": model.log,
                                               "outcomes": model.results.map { $0.outcome.rawValue },
                                               "completed": model.completedCount], to: directory)
                    NSApp.terminate(nil)
                }
                model.start()
                if CommandLine.arguments.contains("--smoke-cancel") {
                    let deadline = Date().addingTimeInterval(10)
                    while model.running && model.progress == nil && Date() < deadline {
                        try? await Task.sleep(for: .milliseconds(20))
                    }
                    model.cancel()
                }
            } else {
                try? NativeSupport.report(["ok": NativeSupport.converter.map { FileManager.default.isExecutableFile(atPath: $0.path) } ?? false], to: directory)
                NSApp.terminate(nil)
            }
        }
    }
}

struct SampleSelectionView: View {
    @ObservedObject var model: ConversionModel
    let request: SampleSelectionRequest
    @State private var selected: [URL: Set<Int>]

    init(model: ConversionModel, request: SampleSelectionRequest) {
        self.model = model
        self.request = request
        _selected = State(initialValue: Dictionary(uniqueKeysWithValues: request.files.map { ($0.input, Set($0.samples)) }))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Choose samples / runs").font(.title2.bold())
            Text(model.inspect ? "Select the WIFF sample identifiers to inspect." :
                 "Each checked sample becomes a separate file. The number comes from the WIFF metadata.")
                .foregroundStyle(SuiteTheme.muted)
            HStack {
                Button("Select All") { selected = Dictionary(uniqueKeysWithValues: request.files.map { ($0.input, Set($0.samples)) }) }
                Button("Deselect All") { selected = Dictionary(uniqueKeysWithValues: request.files.map { ($0.input, Set<Int>()) }) }
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    ForEach(request.files) { file in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(file.input.lastPathComponent).font(.headline).textSelection(.enabled)
                            Text((file.input.deletingLastPathComponent().path as NSString).abbreviatingWithTildeInPath)
                                .font(.caption).foregroundStyle(SuiteTheme.muted).lineLimit(1).truncationMode(.middle)
                            ForEach(file.samples, id: \.self) { id in
                                Toggle(isOn: Binding(get: { selected[file.input, default: []].contains(id) }, set: { checked in
                                    if checked { selected[file.input, default: []].insert(id) }
                                    else { selected[file.input, default: []].remove(id) }
                                })) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text("Sample \(id)")
                                        if !model.inspect, let output = model.sampleDestination(for: file.input, id: id) {
                                            Text(output.lastPathComponent).font(.caption).foregroundStyle(SuiteTheme.muted)
                                        }
                                    }
                                }.toggleStyle(.checkbox)
                                .accessibilityLabel("\(file.input.lastPathComponent), Sample \(id)")
                            }
                        }
                        Divider()
                    }
                }.frame(maxWidth: .infinity, alignment: .leading)
            }.frame(maxHeight: 350)
            if request.includesSingleSampleFiles {
                Text("Single-sample files in the queue will also be processed.").font(.caption).foregroundStyle(SuiteTheme.muted)
            }
            HStack {
                Button("Cancel") { model.cancel() }.keyboardShortcut(.cancelAction)
                Spacer()
                Button(model.inspect ? "Inspect Selected" : "Convert Selected") { model.selectSamples(selected) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!request.includesSingleSampleFiles && selected.values.allSatisfy(\.isEmpty))
            }
        }.padding(24).frame(width: 600).fixedSize(horizontal: false, vertical: true)
            .interactiveDismissDisabled()
    }
}
