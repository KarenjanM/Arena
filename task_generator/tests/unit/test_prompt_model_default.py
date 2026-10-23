from __future__ import annotations

import os
import subprocess
import sys


def test_the_prompt_model_is_the_llm_default() -> None:
    environ = {key: value for key, value in os.environ.items() if key != 'ARENA_PROMPT_LLM'}
    script = 'from arena_simulation_setup.utils.llm import DEFAULT_MODEL; from task_generator.tasks.obstacles.prompt.prompt_utils import REMOTE_LM; print(DEFAULT_MODEL, REMOTE_LM)'
    for env, expected in (({}, 'gemini-3.5-flash'), ({'ARENA_PROMPT_LLM': 'local-test-model'}, 'local-test-model')):
        result = subprocess.run([sys.executable, '-c', script], env={**environ, **env}, capture_output=True, text=True, timeout=120, check=False)
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == [expected, expected]
