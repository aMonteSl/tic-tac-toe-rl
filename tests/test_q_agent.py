from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from ttt.agents.q_agent import QAgent, QAgentConfig


def test_q_agent_greedy_action_selection() -> None:
    state = (0,) * 9
    config = QAgentConfig(epsilon=0.0)
    agent = QAgent(config=config, seed=123)
    agent.q_table = {
        ",".join(map(str, state)): {0: 0.1, 2: 0.5, 4: 0.5}
    }

    action = agent.select_action(state, [0, 2, 4], training=True)
    assert action in {2, 4}


def test_q_agent_update_and_persistence(tmp_path: Path) -> None:
    state = (0,) * 9
    next_state = (1,) + (0,) * 8
    config = QAgentConfig(alpha=1.0, gamma=0.0, epsilon=0.0)
    agent = QAgent(config=config)

    agent.update(state, action=0, reward=1.0, next_state=next_state, next_legal_actions=[1], done=True)
    key = ",".join(map(str, state))
    assert agent.q_table[key][0] == pytest.approx(1.0)

    save_path = tmp_path / "q_table.json"
    agent.save(save_path)

    new_agent = QAgent(config=config)
    new_agent.load(save_path)
    assert new_agent.q_table[key][0] == pytest.approx(1.0)
