import AppKit
import SwiftUI
import KmzMacSupport

@main struct KmzSuiteApp: App {
    var body: some Scene {
        WindowGroup {
            LauncherView()
                .suiteTheme()
                .frame(width: 490, height: 540)
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .help) {
                Button("KmzSuite Manual") { NativeSupport.showManual("KmzSuite") }
            }
        }
    }
}

struct LauncherView: View {
    @State private var error: String?
    @State private var refreshed = false
    @State private var smokeStarted = false
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("KmzSuite").font(.largeTitle.bold())
            Text("Choose a tool").foregroundStyle(SuiteTheme.muted)
            ForEach(SuiteTool.allCases) { tool in
                HStack(spacing: 14) {
                    Image(systemName: tool.symbol).font(.title2).foregroundStyle(SuiteTheme.accent).frame(width: 32)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(tool.rawValue).font(.headline)
                        Text(tool.detail).font(.caption).foregroundStyle(SuiteTheme.muted)
                    }
                    Spacer()
                    Button(tool.isAvailable(in: NativeSupport.distribution) ? "Open" : "Unavailable") {
                        launch(tool)
                    }
                    .disabled(!tool.isAvailable(in: NativeSupport.distribution))
                    .accessibilityLabel("Open " + tool.rawValue)
                }
                .padding(10)
                .background(SuiteTheme.paper, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(SuiteTheme.border, lineWidth: 1))
            }
            HStack {
                Text("Keep the five apps together in the distribution folder.")
                    .font(.caption).foregroundStyle(SuiteTheme.muted)
                Spacer()
                Button { refreshed.toggle() } label: { Image(systemName: "arrow.clockwise") }
                    .help("Refresh availability")
            }
            .id(refreshed)
        }
        .padding(28)
        .alert("Could not launch tool", isPresented: Binding(get: { error != nil }, set: { if !$0 { error = nil } })) {
            Button("OK") { error = nil }
        } message: { Text(error ?? "") }
        .task {
            guard !smokeStarted, let folder = NativeSupport.argument("--smoke-output") else { return }
            smokeStarted = true
            try? await Task.sleep(for: .milliseconds(700))
            let directory = URL(fileURLWithPath: folder, isDirectory: true)
            do {
                var opened: [String] = []
                if CommandLine.arguments.contains("--smoke-launch-all") {
                    for tool in SuiteTool.allCases {
                        let target = directory.appendingPathComponent(tool.rawValue)
                        let config = NSWorkspace.OpenConfiguration()
                        config.createsNewApplicationInstance = true
                        config.arguments = ["--smoke-output", target.path]
                        _ = try await NSWorkspace.shared.openApplication(at: tool.application(in: NativeSupport.distribution), configuration: config)
                        opened.append(tool.rawValue)
                    }
                }
                let available = SuiteTool.allCases.filter { $0.isAvailable(in: NativeSupport.distribution) }.map(\.rawValue)
                try NativeSupport.report(["ok": available.count == SuiteTool.allCases.count, "available": available, "launched": opened], to: directory)
            } catch {
                try? NativeSupport.report(["ok": false, "error": error.localizedDescription], to: directory)
            }
            NSApp.terminate(nil)
        }
    }
    private func launch(_ tool: SuiteTool) {
        NSWorkspace.shared.openApplication(at: tool.application(in: NativeSupport.distribution), configuration: .init()) { _, failure in
            if let failure { Task { @MainActor in error = failure.localizedDescription } }
        }
    }
}
