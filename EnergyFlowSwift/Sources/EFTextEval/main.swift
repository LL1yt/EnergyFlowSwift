import Foundation
import Dispatch
import EnergyFlow
import EFCore

// Top-level entry point driving the async run() without using @main
do {
    let sema = DispatchSemaphore(value: 0)
    var exitCode: Int32 = 0
    Task {
        do {
            try await run()
        } catch {
            Logger.shared.error("EFTextEval failed: \(error)", category: Logger.Category.dataset)
            exitCode = 1
        }
        sema.signal()
    }
    sema.wait()
    exit(exitCode)
}
