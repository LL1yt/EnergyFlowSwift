# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project implementing a **3D Cellular Neural Network** inspired by biological brain structures. The system uses cellular automata-like structures arranged in a 3D lattice where each cell runs the same neural network prototype but processes signals from neighboring cells.

### Core Architecture (Three-Module Pipeline)

1. **Module 1: Teacher LLM Encoder** (`data/embedding_loader/`) - ✅ Complete
   - Converts text → semantic embeddings (768D) using 8+ LLM models
   - Supports LLaMA, Mistral, BERT, and other models with smart caching

2. **Module 2: 3D Cubic Core** (`core/`) - 🔄 In Development
   - `cell_prototype/` - Individual "smart cell" (mini neural network)
   - `lattice_3d/` - 3D grid of interconnected cells with I/O placement strategies
   - `signal_propagation/` - Temporal dynamics and signal spreading patterns

3. **Module 3: Lightweight Decoder** (`inference/`) - 📋 Planned
   - Converts processed embeddings → text output with compact architecture

## Key Development Commands

### Running Tests
```bash
# Run individual module tests
python test_lattice_3d_basic.py
python test_embedding_loader_basic.py
python test_signal_propagation.py

# Run specific functionality tests
python test_io_proportional_strategy.py
python test_data_visualization_fixed.py
```

### Main Application
```bash
# Demo mode (default - shows concepts and tests)
python main.py --mode demo

# Training mode (planned)
python main.py --mode train

# Testing mode (planned) 
python main.py --mode test

# Custom configuration
python main.py --config config/custom_config.yaml --debug
```

### Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# Core ML stack: torch, numpy, scipy, scikit-learn
# Visualization: matplotlib, seaborn, plotly, ipywidgets
# Config: pyyaml, python-dotenv
# Development: jupyter, pytest, black, flake8
```

## Architecture Principles

### Biological Inspiration
- **Cellular Uniformity**: All cells use the same neural network weights (like cortical columns)
- **Local Processing**: Each cell only processes signals from immediate 3D neighbors
- **Parameter Efficiency**: Single prototype scaled across entire network vs. millions of unique parameters

### Key Design Patterns
- **Modular Independence**: Each component can be developed/tested separately
- **Configuration-Driven**: YAML-based settings for all components
- **I/O Placement Strategies**: Proportional coverage (7.8-15.6%) vs. full-face approaches
- **Signal Propagation Modes**: Wave, diffusion, and directional patterns

## Configuration System

### Main Config: `config/main_config.yaml`
```yaml
lattice:
  dimensions: [5, 5, 5]  # Start small for testing
cell_prototype:
  state_size: 8          # Cell internal state dimension
  input_size: 768        # Embedding input dimension
  activation: "tanh"
```

### Module-Specific Configs
- `core/cell_prototype/config/cell_prototype.yaml` - Individual cell settings
- `core/lattice_3d/config/default.yaml` - 3D grid with I/O strategies  
- `data/embedding_loader/config/embedding_config.yaml` - LLM integration

## Code Organization

### Core Components (`core/`)
- **CellPrototype**: Mini neural network that processes neighbor signals
- **Lattice3D**: 3D grid container with sophisticated I/O point placement
- **SignalPropagator**: Manages temporal dynamics and wave propagation patterns

### Data Pipeline (`data/`)
- **EmbeddingLoader**: Handles 8+ LLM models with caching and preprocessing
- **EmbeddingReshaper**: Converts 1D embeddings ↔ 3D spatial representations
- **DataVisualization**: Creates 3D plots and training visualizations

### Key Innovations
1. **Proportional I/O Strategy**: Automatic scaling of input/output points (vs. full-face coverage)
2. **Unified Cell Architecture**: Single prototype replicated across 3D space
3. **Multi-Modal Signal Propagation**: Support for different spreading patterns
4. **Real-Time LLM Integration**: Dynamic embedding generation from text

## Testing Patterns

Tests are organized by module and complexity:
- `test_*_basic.py` - Core functionality tests
- `test_*_advanced.py` - Integration and performance tests  
- `test_*_llm.py` - LLM integration tests

Each test file follows the pattern:
```python
def test_imports():
    """Test 1: Component imports"""
    
def test_basic_functionality():
    """Test 2: Core operations"""
    
def test_configuration():
    """Test 3: Config loading"""
```

## Development Status

### Completed Modules (✅)
- Cell prototype with PyTorch integration
- 3D lattice with I/O placement strategies  
- Signal propagation with multiple modes
- LLM embedding loader (8+ models)
- Data visualization system

### In Progress (🔄)
- EmbeddingReshaper (1D↔3D conversion)
- Core embedding processor integration
- Advanced signal propagation patterns

### Current Phase: 2.5 - Core Embedding Processor
Focus on integrating the 3D lattice with embedding processing pipeline.

## Important Notes

- **Language**: Primary documentation is in Russian, but code comments are in English
- **Biological Analogies**: Code comments often reference brain cortex organization
- **Parameter Efficiency**: Core goal is learning one prototype vs. millions of parameters
- **Visualization Heavy**: Extensive use of 3D plots and animations for understanding

## File Structure Patterns

Each module follows consistent organization:
- `README.md` - Module overview and purpose
- `plan.md` - Implementation roadmap with checkboxes
- `meta.md` - Technical details and dependencies  
- `errors.md` - Known issues and solutions
- `diagram.mmd` - Mermaid visual schema
- `examples.md` - Usage examples
- `config/` - YAML configuration files