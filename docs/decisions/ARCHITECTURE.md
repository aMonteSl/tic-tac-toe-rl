# Architecture Decision Records

## Overview

This document captures key architectural decisions for the Tic-Tac-Toe RL project.

## ADR-001: Project Structure

**Date**: 2026-02-05

**Status**: Accepted

**Context**: 
We need a clear, maintainable structure for a reinforcement learning project that will grow over time with multiple agents, training algorithms, and evaluation tools.

**Decision**: 
Organize code into distinct modules under `src/ttt/`:
- `env/` - Game environment (isolated from agents)
- `agents/` - Agent implementations
- `training/` - Training infrastructure
- `evaluation/` - Metrics and evaluation tools
- `rendering/` - Visualization (separate from core logic)
- `utils/` - Shared utilities

**Consequences**:
- Clear separation of concerns
- Easy to add new agents without modifying environment
- Training and evaluation can evolve independently
- Rendering is optional and doesn't interfere with core logic

**Rationale**:
This structure follows common RL project patterns and supports independent development of each component.

---

## ADR-002: Environment as Independent Module

**Date**: 2026-02-05

**Status**: Accepted

**Context**: 
The Tic-Tac-Toe environment needs to be reusable across different agent implementations and training approaches.

**Decision**: 
Implement the environment as a standalone module that:
- Has no dependencies on specific agent implementations
- Provides a clean interface for state, actions, and rewards
- Handles all game logic internally
- Can be used for both training and evaluation

**Consequences**:
- Agents are decoupled from environment implementation
- Environment can be tested independently
- Easy to swap or modify without affecting agents
- Supports multiple agent types using the same environment

**Rationale**:
Following OpenAI Gym-like patterns ensures flexibility and testability.

---

## ADR-003: Self-Play Training Approach

**Date**: 2026-02-05

**Status**: Accepted

**Context**: 
Agents need to learn through experience, and Tic-Tac-Toe is well-suited for self-play training.

**Decision**: 
Primary training method will be self-play where:
- Agents play against themselves or previous versions
- Experience is generated from these games
- Training uses this experience to improve

**Consequences**:
- No need for human-labeled data
- Agents learn strategies through exploration
- Requires careful curriculum design to avoid local optima
- May need opponent pool management

**Rationale**:
Self-play has proven effective for board games and eliminates need for external data sources.

---

## ADR-004: Multiple Agent Support

**Date**: 2026-02-05

**Status**: Accepted

**Context**: 
We want to experiment with different RL algorithms and compare their performance.

**Decision**: 
Support multiple agent types through a common interface:
- Base agent class/interface defining required methods
- Specific implementations for different algorithms
- All agents compatible with the same environment and training loop

**Consequences**:
- Easy to add new algorithms
- Can compare different approaches
- Consistent API across all agents
- Some algorithms may need algorithm-specific training code

**Rationale**:
Flexibility to experiment with different RL approaches while maintaining code consistency.

---

## ADR-005: Pygame for Rendering

**Date**: 2026-02-05

**Status**: Accepted

**Context**: 
Need a simple, cross-platform way to visualize games and potentially allow human interaction.

**Decision**: 
Use pygame for all rendering and visualization:
- Simple 2D graphics for Tic-Tac-Toe board
- Cross-platform compatibility
- Active community and good documentation
- Lightweight dependency

**Consequences**:
- Additional dependency on pygame
- Rendering module separate from core logic
- Can build interactive demos easily
- May need to handle pygame event loop carefully

**Rationale**:
Pygame is simple, well-suited for 2D board games, and widely used in Python game development.

---

## ADR-006: NumPy for Numerical Operations

**Date**: 2026-02-05

**Status**: Accepted

**Context**: 
Need efficient numerical operations for state representation, rewards, and potentially neural networks.

**Decision**: 
Use NumPy as the primary numerical library:
- State representation using NumPy arrays
- Efficient batch operations
- Standard library in ML/RL projects
- Good integration with other libraries if needed later

**Consequences**:
- Dependency on NumPy
- Efficient numerical operations
- May use additional libraries (like PyTorch/TensorFlow) for deep learning later
- NumPy arrays as standard data format

**Rationale**:
NumPy is the de facto standard for numerical computing in Python and provides the foundation we need.

---

## Design Principles

### Modularity
Each component should be independently testable and replaceable.

### Simplicity
Start with simple implementations and add complexity only when needed.

### Extensibility  
Design interfaces that make it easy to add new agents, training methods, or evaluation metrics.

### Testability
All core logic should be easily testable without requiring rendering or long training runs.

## Future Considerations

- Deep learning frameworks (PyTorch or TensorFlow) if implementing neural network agents
- Distributed training support for scaling up
- Model checkpointing and versioning
- Advanced logging and experiment tracking (e.g., TensorBoard, Weights & Biases)
