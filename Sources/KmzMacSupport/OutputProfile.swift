import Foundation

/// Names are UI-only; scientific settings belong to the versioned Python contract.
public enum OutputProfile: String, CaseIterable, Identifiable {
    case full, sparse
    public var id: String { rawValue }
    public var title: String { self == .full ? "Full Profile" : "Sparse Profile v3" }
    public var detail: String { self == .full ? "Complete, authoritative profile mzML" : "Smaller, lossy MS1 representation for inspection" }

    public func suggestedOutputURL(for input: URL, massCorrection: Bool = false) -> URL {
        input.deletingPathExtension().appendingPathExtension((massCorrection ? "mc." : "") + (self == .full ? "mzML" : "sp.mzML"))
    }

    public func outputURL(_ url: URL) -> URL {
        var name = url.lastPathComponent
        if name.lowercased().hasSuffix(".mzml") { name = String(name.dropLast(5)) }
        if name.lowercased().hasSuffix(".sp") { name = String(name.dropLast(3)) }
        while name.lowercased().hasSuffix(".mzml") { name = String(name.dropLast(5)) }
        return url.deletingLastPathComponent().appendingPathComponent(name + (self == .full ? ".mzML" : ".sp.mzML"))
    }

    public func sampleOutputURL(_ url: URL, sampleID: Int) -> URL {
        let name = url.lastPathComponent
        let suffix = name.lowercased().hasSuffix(".sp.mzml") ? ".sp.mzML" : ".mzML"
        return url.deletingLastPathComponent().appendingPathComponent(String(name.dropLast(suffix.count)) + "_\(sampleID)" + suffix)
    }
}
