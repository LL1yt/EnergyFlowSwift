import Foundation
import os

// MARK: - Unified logging via os.Logger (Apple best practice)
public enum LogLevel: Int {
    case error = 0
    case warn = 1
    case info = 2
    case debug = 3
}

public final class Logger: @unchecked Sendable {
    public static let shared = Logger()
    private init() {}

    // Toggle and level gate for runtime control in research/dev
    public var enabled: Bool = true
    public var level: LogLevel = .info

    // Subsystem/category allow grouping logs in Console.app
    private let subsystem = "com.aa.energyflowswift"
    private let defaultLogger = os.Logger(subsystem: "com.aa.energyflowswift", category: "default")
    private var categoryLoggers: [String: os.Logger] = [:]

    private func logger(category: String?) -> os.Logger {
        guard let category = category else { return defaultLogger }
        if let l = categoryLoggers[category] { return l }
        let l = os.Logger(subsystem: subsystem, category: category)
        categoryLoggers[category] = l
        return l
    }

    public func log(_ level: LogLevel, _ message: @autoclosure () -> String,
                    category: String? = nil,
                    file: StaticString = #file, function: StaticString = #function, line: UInt = #line) {
        guard enabled, level.rawValue <= self.level.rawValue else { return }
        let fname = ("\(file)" as NSString).lastPathComponent
        let text = "\(fname):\(line) \(function) — \(message())"
        let l = logger(category: category)
        switch level {
        case .error:
            l.error("\(text, privacy: .public)")
        case .warn:
            l.warning("\(text, privacy: .public)")
        case .info:
            l.info("\(text, privacy: .public)")
        case .debug:
            l.debug("\(text, privacy: .public)")
        }
    }

    public func error(_ msg: @autoclosure () -> String, category: String? = nil, file: StaticString = #file, function: StaticString = #function, line: UInt = #line) {
        log(.error, msg(), category: category, file: file, function: function, line: line)
    }
    public func warn(_ msg: @autoclosure () -> String, category: String? = nil, file: StaticString = #file, function: StaticString = #function, line: UInt = #line) {
        log(.warn, msg(), category: category, file: file, function: function, line: line)
    }
    public func info(_ msg: @autoclosure () -> String, category: String? = nil, file: StaticString = #file, function: StaticString = #function, line: UInt = #line) {
        log(.info, msg(), category: category, file: file, function: function, line: line)
    }
    public func debug(_ msg: @autoclosure () -> String, category: String? = nil, file: StaticString = #file, function: StaticString = #function, line: UInt = #line) {
        log(.debug, msg(), category: category, file: file, function: function, line: line)
    }
}
