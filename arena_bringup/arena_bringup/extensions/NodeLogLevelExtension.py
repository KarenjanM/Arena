from __future__ import annotations

import functools
import json
import logging
import os
import re
from typing import Literal

import launch
import launch.utilities
import yaml
from launch_ros.actions import Node as NodeAction
from launch_ros.actions.node import NodeActionExtension
from launch_ros.utilities import plugin_support

LogLevel = Literal['debug', 'info', 'warn', 'error', 'fatal']
_LEVELS: frozenset[str] = frozenset({'debug', 'info', 'warn', 'error', 'fatal'})

_RULES_KEY = 'NodeLogLevelExtension_log_level'
_DEFAULT_PATTERN = '**/*'
_DOUBLESTAR = '\x00DOUBLESTAR\x00'


def _validate_level(level: str) -> str:
    if level not in _LEVELS:
        raise ValueError(f'invalid log level {level!r}; expected one of {sorted(_LEVELS)}')
    return level


def _parse_inline(spec: str) -> list[tuple[str, str]]:
    inner = spec.strip()
    if not (inner.startswith('{') and inner.endswith('}')):
        raise ValueError(f'inline log_level spec must be wrapped in {{...}}: {spec!r}')
    inner = inner[1:-1].strip()
    if not inner:
        raise ValueError('empty log_level spec')

    entries = [e.strip() for e in inner.split(',') if e.strip()]
    rules: list[tuple[str, str]] = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        if ':' in entry:
            pattern, _, level = entry.rpartition(':')
            rules.append((pattern.strip(), _validate_level(level.strip())))
        else:
            if not is_last:
                raise ValueError(f'bare log_level {entry!r} only allowed as last entry; got at position {i} of {entries}')
            rules.append((_DEFAULT_PATTERN, _validate_level(entry)))
    return rules


def _parse_yaml_file(path: str) -> list[tuple[str, str]]:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f'log_level YAML must be a mapping: {path}')
    rules: list[tuple[str, str]] = []
    for entry in doc.get('rules') or []:
        if not isinstance(entry, dict) or 'match' not in entry or 'level' not in entry:
            raise ValueError(f'each log_level rule must be {{match, level}}: {entry!r}')
        rules.append((str(entry['match']), _validate_level(str(entry['level']))))
    if 'default' in doc:
        rules.append((_DEFAULT_PATTERN, _validate_level(str(doc['default']))))
    return rules


def _parse_rule_list(inner: str) -> list[tuple[str, str]]:
    entries = [e.strip() for e in inner.split(',') if e.strip()]
    rules: list[tuple[str, str]] = []
    for entry in entries:
        if ':' not in entry:
            raise ValueError(f'rule {entry!r} in [...] must be <glob>:<level>')
        pattern, _, level = entry.rpartition(':')
        rules.append((pattern.strip(), _validate_level(level.strip())))
    return rules


Op = Literal['replace', 'prepend', 'append']


def parse_log_level_spec(value: str) -> tuple[Op, list[tuple[str, str]]]:
    value = value.strip()
    if not value:
        return 'replace', []
    if value.startswith('+[') and value.endswith(']'):
        return 'prepend', _parse_rule_list(value[2:-1])
    if value.startswith('[') and value.endswith(']+'):
        return 'append', _parse_rule_list(value[1:-2])
    if value.startswith('{'):
        return 'replace', _parse_inline(value)
    if value in _LEVELS:
        return 'replace', [(_DEFAULT_PATTERN, value)]
    if os.path.isfile(value):
        return 'replace', _parse_yaml_file(value)
    raise ValueError(f'log_level value {value!r} is not a known level, inline {{...}} spec, merge form +[...] / [...]+, or path to an existing YAML file')


@functools.lru_cache(maxsize=256)
def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    pattern = pattern.lstrip('/')
    if not pattern:
        pattern = '**'

    out: list[str] = []
    for seg in pattern.split('/'):
        if seg == '**':
            out.append(_DOUBLESTAR)
            continue
        chunk: list[str] = []
        for ch in seg:
            if ch == '*':
                chunk.append('[^/]*')
            elif ch == '?':
                chunk.append('[^/]')
            else:
                chunk.append(re.escape(ch))
        out.append(''.join(chunk))

    regex = '/'.join(out)
    regex = regex.replace(f'/{_DOUBLESTAR}/', '/(?:.*/)?')
    regex = regex.replace(f'{_DOUBLESTAR}/', '(?:.*/)?')
    regex = regex.replace(f'/{_DOUBLESTAR}', '(?:/.*)?')
    regex = regex.replace(_DOUBLESTAR, '.*')
    return re.compile(f'^{regex}$')


def active_spec(context: launch.LaunchContext) -> str:
    """Inline spec of the rule set active in ``context``, empty when none is set."""
    raw = context.launch_configurations.get(_RULES_KEY)
    rules = json.loads(raw) if raw else []
    return '{' + ', '.join(f'{pattern}:{level}' for pattern, level in rules) + '}' if rules else ''


def _match_level(rules: list[tuple[str, str]], fqn: str) -> str | None:
    target = fqn.lstrip('/')
    for pattern, level in rules:
        if _glob_to_regex(pattern).match(target):
            return level
    return None


class NodeLogLevelExtension(NodeActionExtension):
    NAME = "NodeLogLevelExtension"
    EXTENSION_POINT_VERSION = '0.1'

    def __init__(self) -> None:
        super(NodeActionExtension, self).__init__()
        plugin_support.satisfies_version(self.EXTENSION_POINT_VERSION, '^0.1')

    def prepare_for_execute(
        self,
        context: launch.LaunchContext,
        ros_specific_arguments: dict,
        node_action: NodeAction | None,
    ) -> tuple[list, dict]:
        raw = context.launch_configurations.get(_RULES_KEY)
        if raw is None:
            return [], ros_specific_arguments

        rules: list[tuple[str, str]] = [(p, lvl) for p, lvl in json.loads(raw)]
        if not rules:
            return [], ros_specific_arguments

        fqn = node_action.node_name if node_action is not None else ''
        level = _match_level(rules, fqn)
        if level is None:
            return [], ros_specific_arguments

        return [
            [launch.substitutions.TextSubstitution(text='--log-level')],
            [launch.substitutions.TextSubstitution(text=level)],
        ], ros_specific_arguments


class SetGlobalLogLevelAction(launch.Action):
    LOGLEVEL = LogLevel
    _spec: str | launch.Substitution
    _base: str
    _fallback: bool

    def __init__(self, spec: str | launch.Substitution, *, base: str = 'warn', fallback: bool = False, **kwargs: launch.Condition | None) -> None:
        super().__init__(**kwargs)
        self._spec = spec
        self._base = _validate_level(base)
        self._fallback = fallback

    @classmethod
    def str_to_level(cls, log_level: str) -> int:
        return {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warn': logging.WARN,
            'error': logging.ERROR,
            'fatal': logging.FATAL,
        }.get(log_level, logging.NOTSET)

    def execute(self, context: launch.LaunchContext) -> None:
        spec_subs = launch.utilities.normalize_to_list_of_substitutions([self._spec])
        spec = launch.utilities.perform_substitutions(context, spec_subs)
        op, rules = parse_log_level_spec(spec)
        existing_raw = context.launch_configurations.get(_RULES_KEY)
        existing: list[tuple[str, str]] = [(p, lvl) for p, lvl in json.loads(existing_raw)] if existing_raw else []
        if op != 'replace' and not existing:
            existing = [(_DEFAULT_PATTERN, self._base)]
        if op != 'replace' and self._fallback:
            at = next((i for i, (pattern, _) in enumerate(existing) if pattern == _DEFAULT_PATTERN), len(existing))
            merged = existing[:at] + rules + existing[at:]
        elif op == 'prepend':
            merged = rules + existing
        elif op == 'append':
            merged = existing + rules
        else:
            merged = rules
        context.launch_configurations[_RULES_KEY] = json.dumps(merged)
