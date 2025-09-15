# EnergyFlowSwift

Swift/MPS/Metal port scaffolding for the energy_flow research project.

- Targets
  - EFCore: tensor and low-level ops (starts CPU-only)
  - PyTorchSwift: small analogs of PyTorch modules (Embedding, Linear, LayerNorm, Activations)
  - EnergyFlow: architecture modules starting with TextBridge/TextToCubeEncoder

Build and test
- swift build
- swift test

See docs/Swift_MPS_Port_Plan.md for the roadmap and Phase 1 details.
