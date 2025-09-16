import Foundation

public struct EnergyConfig {
    public var latticeWidth: Int
    public var latticeHeight: Int
    public var latticeDepth: Int
    public init(width: Int, height: Int, depth: Int) {
        self.latticeWidth = width
        self.latticeHeight = height
        self.latticeDepth = depth
    }
    public var surfaceDim: Int { latticeWidth * latticeHeight }
}

public func createDebugConfig() -> EnergyConfig {
    EnergyConfig(width: 16, height: 16, depth: 10)
}

public func createExperimentConfig() -> EnergyConfig {
    EnergyConfig(width: 50, height: 50, depth: 20)
}

public func createOptimizedConfig() -> EnergyConfig {
    EnergyConfig(width: 100, height: 100, depth: 50)
}
