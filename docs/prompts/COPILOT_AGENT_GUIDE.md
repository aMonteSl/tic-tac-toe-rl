# Copilot Agent Guide

This guide helps GitHub Copilot agents understand the project structure and development workflow for the Tic-Tac-Toe RL project.

## Project Overview

This is a reinforcement learning project focused on training agents to play Tic-Tac-Toe through self-play. The project emphasizes clean architecture, modularity, and extensibility.

## Code Organization

### src/ttt/env/
Contains the Tic-Tac-Toe game environment implementation:
- Game state representation
- Action validation
- Win/draw detection
- State transitions

### src/ttt/agents/
Agent implementations:
- Base agent interface
- Random agent (baseline)
- Q-Learning agent
- Deep Q-Network (DQN) agent
- Policy gradient agents

### src/ttt/training/
Training infrastructure:
- Self-play mechanisms
- Training loops
- Experience replay
- Checkpoint management

### src/ttt/evaluation/
Evaluation tools:
- Agent vs agent matches
- Performance metrics
- Statistical analysis
- Tournament systems

### src/ttt/rendering/
Visualization with pygame:
- Game board rendering
- Interactive play mode
- Training visualization

### src/ttt/utils/
Utility functions:
- Configuration management
- Logging utilities
- Helper functions

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Write docstrings for all public classes and functions
- Keep functions focused and modular

### Testing
- Write unit tests for new functionality
- Use pytest for test framework
- Aim for high test coverage
- Include integration tests for key workflows

### Documentation
- Update README.md for user-facing changes
- Document architecture decisions in docs/decisions/
- Keep ROADMAP.md updated with progress

## Key Principles

1. **Modularity**: Each component should have a single, well-defined responsibility
2. **Extensibility**: Easy to add new agents or modify existing ones
3. **Testability**: Code should be easy to test in isolation
4. **Clarity**: Code should be self-documenting with clear naming

## Common Tasks

### Adding a New Agent
1. Create agent class in src/ttt/agents/
2. Inherit from base agent interface
3. Implement required methods (select_action, update, etc.)
4. Add tests in tests/
5. Update documentation

### Modifying the Environment
1. Update environment code in src/ttt/env/
2. Ensure backward compatibility or update all dependent code
3. Add/update tests
4. Document changes in ARCHITECTURE.md if significant

### Adding Training Features
1. Implement in src/ttt/training/
2. Ensure it works with existing agents
3. Add configuration options
4. Document usage

## Debugging Tips

- Use logging extensively during training
- Visualize agent behavior with rendering module
- Check state transitions carefully in environment
- Verify reward signals are correct

## Resources

- See ARCHITECTURE.md for system design details
- Check ROADMAP.md for current priorities
- Review existing tests for examples
