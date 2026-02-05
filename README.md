# Tic-Tac-Toe RL (Self-Play)

A reinforcement learning project where an agent learns to play Tic-Tac-Toe through self-play using Python, with a clean and extensible environment design.

## Requirements

- Python 3.11+
- Dependencies: numpy, pygame

## Setup

### Creating Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Installing Dependencies

After activating the virtual environment, install required packages:

```bash
pip install -r requirements.txt
```

## Project Structure

```
tic-tac-toe-rl/
├── src/ttt/              # Main source code
│   ├── agents/           # RL agents and their implementations
│   ├── env/              # Tic-Tac-Toe environment
│   ├── rendering/        # Game visualization with pygame
│   ├── training/         # Training loops and algorithms
│   ├── evaluation/       # Agent evaluation and metrics
│   └── utils/            # Utility functions and helpers
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
│   ├── prompts/          # AI assistant guides and prompts
│   ├── decisions/        # Architecture decision records
│   └── notes/            # Development notes and planning
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Roadmap

See [docs/notes/ROADMAP.md](docs/notes/ROADMAP.md) for the complete development roadmap.

### Planned Features

1. **Environment**: Complete Tic-Tac-Toe game logic with state management
2. **Agents**: Multiple RL algorithms (Q-Learning, Deep Q-Network, Policy Gradient)
3. **Self-Play**: Training agents through self-play mechanisms
4. **Visualization**: Interactive game rendering with pygame
5. **Evaluation**: Comprehensive metrics and agent comparison tools

## Development

This project is structured to support clean, modular development of reinforcement learning agents. Each component is designed to be extensible and testable.

For architecture details, see [docs/decisions/ARCHITECTURE.md](docs/decisions/ARCHITECTURE.md).

## License

TBD
