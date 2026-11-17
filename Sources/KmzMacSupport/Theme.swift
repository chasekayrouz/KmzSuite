import SwiftUI

/// Native counterparts of kmzcore.theme; one restrained light palette for the suite.
public enum SuiteTheme {
    public static let background = Color(red: 244 / 255, green: 246 / 255, blue: 248 / 255)
    public static let paper = Color.white
    public static let ink = Color(red: 36 / 255, green: 51 / 255, blue: 73 / 255)
    public static let muted = Color(red: 82 / 255, green: 100 / 255, blue: 119 / 255)
    public static let border = Color(red: 204 / 255, green: 214 / 255, blue: 222 / 255)
    public static let accent = Color(red: 38 / 255, green: 92 / 255, blue: 81 / 255)
    public static let disabledText = Color(red: 116 / 255, green: 129 / 255, blue: 141 / 255)
    public static let disabledPaper = Color(red: 237 / 255, green: 240 / 255, blue: 243 / 255)
    public static let selection = Color(red: 220 / 255, green: 238 / 255, blue: 231 / 255)
}

public struct SuiteButtonStyle: ButtonStyle {
    private let prominent: Bool
    public init(prominent: Bool = false) { self.prominent = prominent }
    public func makeBody(configuration: Configuration) -> some View {
        Surface(configuration: configuration, prominent: prominent)
    }

    private struct Surface: View {
        let configuration: ButtonStyleConfiguration
        let prominent: Bool
        @Environment(\.isEnabled) private var enabled
        @Environment(\.isFocused) private var focused
        @State private var hovered = false
        var body: some View {
            let active = enabled && (hovered || focused || configuration.isPressed)
            configuration.label
                .foregroundStyle(enabled ? (prominent ? SuiteTheme.paper : SuiteTheme.ink) : SuiteTheme.disabledText)
                .padding(.horizontal, 9).padding(.vertical, 4)
                .background(enabled ? (prominent ? SuiteTheme.accent : active ? SuiteTheme.selection : SuiteTheme.paper) : SuiteTheme.disabledPaper,
                            in: RoundedRectangle(cornerRadius: 6))
                .overlay(RoundedRectangle(cornerRadius: 6)
                    .stroke(active || (prominent && enabled) ? SuiteTheme.accent : SuiteTheme.border,
                            lineWidth: focused ? 2 : 1))
                .contentShape(RoundedRectangle(cornerRadius: 6))
                .onHover { hovered = $0 }
        }
    }
}

public struct SuitePanelStyle: GroupBoxStyle {
    public init() {}
    public func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            configuration.label
            configuration.content
        }
        .padding(10)
        .background(SuiteTheme.paper, in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(SuiteTheme.border, lineWidth: 1))
    }
}

public extension View {
    func suiteTheme() -> some View {
        self.preferredColorScheme(.light)
            .tint(SuiteTheme.accent).accentColor(SuiteTheme.accent)
            .foregroundStyle(SuiteTheme.ink)
            .background(SuiteTheme.background)
            .groupBoxStyle(SuitePanelStyle())
            .buttonStyle(SuiteButtonStyle())
    }
}
