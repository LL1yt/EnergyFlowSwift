// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "EnergyFlowSwift",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "EFCore", targets: ["EFCore"]),
        .library(name: "PyTorchSwift", targets: ["PyTorchSwift"]),
        .library(name: "EnergyFlow", targets: ["EnergyFlow"]),
    ],
    targets: [
        .target(
            name: "EFCore",
            dependencies: [],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph")
            ]
        ),
        .target(
            name: "PyTorchSwift",
            dependencies: ["EFCore"]
        ),
        .target(
            name: "EnergyFlow",
            dependencies: ["EFCore", "PyTorchSwift"]
        ),
        .testTarget(
            name: "EnergyFlowSwiftTests",
            dependencies: ["EnergyFlow"]
        ),
    ]
)
