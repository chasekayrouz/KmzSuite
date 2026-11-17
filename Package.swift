// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "KmzSuite",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "KmzSuite", targets: ["KmzSuite"]),
        .executable(name: "KmzConvert", targets: ["KmzConvert"])
    ],
    targets: [
        .target(name: "KmzMacSupport", path: "Sources/KmzMacSupport"),
        .executableTarget(name: "KmzSuite", dependencies: ["KmzMacSupport"], path: "Apps/KmzSuite"),
        .executableTarget(name: "KmzConvert", dependencies: ["KmzMacSupport"], path: "Apps/KmzConvert"),
        .testTarget(name: "KmzMacSupportTests", dependencies: ["KmzMacSupport"], path: "Tests/Native"),
        .testTarget(name: "KmzConvertTests", dependencies: ["KmzConvert"], path: "Tests/ConvertNative")
    ],
    swiftLanguageModes: [.v5]
)
