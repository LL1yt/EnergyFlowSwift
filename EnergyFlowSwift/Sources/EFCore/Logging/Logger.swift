import Foundation
import os

// MARK: - Unified logging via os.Logger (Apple best practice)
public enum LogLevel: Int {
    case error = 0
    case warn = 1
    case info = 2
    case info1 = 3   // informative, less verbose than debug, more than info
    case debug = 4
}

public final class Logger: @unchecked Sendable {
    public static let shared = Logger()
    private init() {
        // Read environment for default behavior in CLI/testing
        let env = ProcessInfo.processInfo.environment
        if let s = env["EF_LOG_LEVEL"] {
            self.level = Logger.parseLevel(s)
        }
        if let s = env["EF_LOG_ENABLED"] {
            self.enabled = (s == "1" || s.lowercased() == "true" || s.lowercased() == "yes")
        }
        if let s = env["EF_LOG_STDOUT"] {
            self.mirrorToStdout = (s == "1" || s.lowercased() == "true" || s.lowercased() == "yes")
        }
    }

    // Common categories to avoid typos
    public struct Category {
        public static let textBridge = "TextBridge"
        public static let training   = "Training"
        public static let dataset    = "Dataset"
    }

    // Toggle and level gate for runtime control in research/dev
    public var enabled: Bool = true
    public var level: LogLevel = .info
    public var mirrorToStdout: Bool = false // also print to stdout for CLI/testing

    // Subsystem/category allow grouping logs in Console.app
    private let subsystem = "com.aa.energyflowswift"
    private let defaultLogger = os.Logger(subsystem: "com.aa.energyflowswift", category: "default")
    private var categoryLoggers: [String: os.Logger] = [:]

    private static func parseLevel(_ s: String) -> LogLevel {
        switch s.lowercased() {
        case "error": return .error
        case "warn", "warning": return .warn
        case "info": return .info
        case "info1", "info_1": return .info1
        case "debug": return .debug
        default: return .info
        }
    }

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
        case .info1:
            l.info("\(text, privacy: .public)")
        case .debug:
            l.debug("\(text, privacy: .public)")
        }
        if mirrorToStdout {
            let lvl: String
            switch level {
            case .error: lvl = "ERROR"
            case .warn:  lvl = "WARN"
            case .info:  lvl = "INFO"
            case .info1: lvl = "INFO1"
            case .debug: lvl = "DEBUG"
            }
            let cat = category ?? "default"
            Swift.print("[\(lvl)][\(cat)] \(text)")
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
    public func info1(_ msg: @autoclosure () -> String, category: String? = nil, file: StaticString = #file, function: StaticString = #function, line: UInt = #line) {
        log(.info1, msg(), category: category, file: file, function: function, line: line)
    }
}
