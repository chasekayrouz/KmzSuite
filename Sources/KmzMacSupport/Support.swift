import AppKit
import Foundation

public enum SuiteTool: String, CaseIterable, Identifiable {
    case viewer = "KmzViewer", viewer2 = "Kmz2Viewer", convert = "KmzConvert", comet = "KmzCoMet"
    public var id: String { rawValue }
    public var detail: String {
        switch self {
        case .viewer: return "Chromatograms and MS1 spectra"
        case .viewer2: return "MS2 scans and mass differences"
        case .convert: return "SCIEX WIFF and Agilent .d to mzML"
        case .comet: return "Compare MS1 features across acquisitions"
        }
    }
    public var symbol: String {
        switch self {
        case .viewer: return "waveform.path"
        case .viewer2: return "chart.bar.xaxis"
        case .convert: return "arrow.triangle.2.circlepath"
        case .comet: return "tablecells"
        }
    }
    public func application(in directory: URL) -> URL {
        directory.appendingPathComponent(rawValue + ".app", isDirectory: true)
    }
    public func isAvailable(in directory: URL) -> Bool {
        FileManager.default.isExecutableFile(atPath: application(in: directory)
            .appendingPathComponent("Contents/MacOS/" + rawValue).path)
    }
}

public struct ConversionProgress: Equatable {
    public let done: Int
    public let total: Int
    public static func parse(_ text: String) -> ConversionProgress? {
        let pattern = #"Converting (\d+)/(\d+) spectra"#
        guard let expression = try? NSRegularExpression(pattern: pattern),
              let match = expression.matches(in: text, range: NSRange(text.startIndex..., in: text)).last,
              let first = Range(match.range(at: 1), in: text),
              let second = Range(match.range(at: 2), in: text),
              let done = Int(text[first]), let total = Int(text[second]),
              total > 0, done >= 0, done <= total else { return nil }
        return ConversionProgress(done: done, total: total)
    }
}

public enum NativeSupport {
    /// The suite's tools are the launcher's siblings in the KmzSuite folder.
    public static func toolDirectory(for bundle: Bundle) -> URL {
        bundle.bundleURL.deletingLastPathComponent()
    }
    public static var distribution: URL { toolDirectory(for: .main) }
    public static var converter: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("Converter/kmzconvert")
    }
    public static func argument(_ name: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: name), args.indices.contains(i + 1) else { return nil }
        return args[i + 1]
    }
    @MainActor public static func showManual(_ name: String) {
        if let file = Bundle.main.resourceURL?.appendingPathComponent("docs/" + name + ".html") {
            NSWorkspace.shared.open(file)
        }
    }
    public static func report(_ value: [String: Any], to directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("result.json"))
    }
}
