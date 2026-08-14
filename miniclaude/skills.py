"""Lightweight on-demand skills loaded from ``skills/<name>/SKILL.md`` files.

A skill is procedural guidance (when to use it, which tools to prefer, how to
verify) that is injected into the system instructions only when it matches the
task. Skills are not executable capabilities: they are realized through the
same registered tools the model already has.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_META_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
_KEYWORD_RE = re.compile(r"[a-z][a-z0-9_\-]+")


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """Parsed skill metadata plus the body used for instruction injection."""

    name: str
    description: str
    when_to_use: str
    version: str
    content: str
    tools: tuple[str, ...] = ()
    path: Path | None = None

    def keywords(self) -> tuple[str, ...]:
        text = f"{self.when_to_use} {self.description}".lower()
        return tuple(dict.fromkeys(_tokens(text)))

    @classmethod
    def from_file(cls, path: Path) -> "SkillSpec":
        raw = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER_RE.match(raw)
        if match is None:
            raise ValueError(f"skill file has no front matter: {path}")
        meta: dict[str, str] = {}
        for line in match.group(1).splitlines():
            meta_match = _META_RE.match(line.strip())
            if meta_match:
                meta[meta_match.group(1)] = meta_match.group(2).strip().strip("\"'")
        name = meta.get("name", "").strip()
        description = meta.get("description", "").strip()
        when_to_use = meta.get("when_to_use", "").strip()
        version = meta.get("version", "1.0.0").strip()
        tools = tuple(
            tool.strip()
            for tool in meta.get("tools", "").split(",")
            if tool.strip()
        )
        if not name or not description or not when_to_use:
            raise ValueError(f"skill metadata is incomplete: {path}")
        body = raw[match.end():].strip()
        if not body:
            raise ValueError(f"skill body is empty: {path}")
        return cls(
            name=name,
            description=description,
            when_to_use=when_to_use,
            version=version,
            content=body,
            tools=tools,
            path=path,
        )


_STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in is it its of on or that
    the this to was were will with your you all can not any
    """.split()
)


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9_\-]+", lowered)
        if len(token) >= 3 and token not in _STOPWORDS
    ]


class SemanticSkillSelector:
    """TF-IDF cosine retrieval over the skill corpus.

    ``fit`` builds term vectors from each skill's metadata and body; ``score``
    returns a cosine similarity per skill for a task. Pure ``math``/stdlib, so
    routing stays offline and deterministic.
    """

    def __init__(self, min_cosine: float = 0.15):
        if min_cosine < 0:
            raise ValueError("min_cosine must not be negative")
        self.min_cosine = min_cosine
        self._idf: dict[str, float] = {}
        self._vectors: dict[str, dict[str, float]] = {}

    def fit(self, skills: Iterable[SkillSpec]) -> "SemanticSkillSelector":
        documents = {
            spec.name: Counter(
                _tokens(
                    f"{spec.when_to_use} {spec.description} {spec.content}"
                )
            )
            for spec in skills
        }
        document_frequency: Counter[str] = Counter()
        for counts in documents.values():
            document_frequency.update(counts.keys())
        total = len(documents)
        self._idf = {
            term: math.log((total + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._vectors = {}
        for name, counts in documents.items():
            vector = {
                term: count * self._idf[term]
                for term, count in counts.items()
            }
            norm = math.sqrt(sum(value * value for value in vector.values()))
            self._vectors[name] = (
                {term: value / norm for term, value in vector.items()}
                if norm
                else {}
            )
        return self

    def score(self, task: str) -> dict[str, float]:
        """Cosine similarity from a task to every fitted skill."""
        query = Counter(_tokens(task))
        query_vector = {
            term: count * self._idf.get(term, 0.0)
            for term, count in query.items()
        }
        norm = math.sqrt(
            sum(value * value for value in query_vector.values())
        )
        if not norm:
            return {name: 0.0 for name in self._vectors}
        query_vector = {
            term: value / norm for term, value in query_vector.items()
        }
        return {
            name: sum(
                query_vector.get(term, 0.0) * value
                for term, value in vector.items()
            )
            for name, vector in self._vectors.items()
        }

    def best(self, task: str) -> list[tuple[str, float]]:
        return sorted(
            (
                (name, cosine)
                for name, cosine in self.score(task).items()
                if cosine >= self.min_cosine
            ),
            key=lambda item: (-item[1], item[0]),
        )


class SkillRegistry:
    """Discovers skills on disk and selects matching skills for a task."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else None
        self._skills: dict[str, SkillSpec] = {}
        self._semantic: SemanticSkillSelector | None = None
        if self.root is not None:
            self.discover(self.root)

    def discover(self, root: str | Path | None = None) -> list[str]:
        base = Path(root) if root is not None else self.root
        if base is None or not base.is_dir():
            return []
        found: list[str] = []
        for skill_file in sorted(base.glob("*/SKILL.md")):
            spec = SkillSpec.from_file(skill_file)
            self._skills[spec.name] = spec
            found.append(spec.name)
        self._semantic = None
        return found

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> SkillSpec | None:
        return self._skills.get(name)

    def select(
        self,
        task: str,
        top_k: int = 1,
        mode: str = "hybrid",
    ) -> list[SkillSpec]:
        """Select up to ``top_k`` skills for a task.

        Modes:
        - ``keyword``: deterministic keyword scoring (previous behavior).
        - ``semantic``: pure TF-IDF cosine retrieval above a threshold.
        - ``hybrid`` (default): keyword recall gates candidates, then cosine
          reranks them; unrelated tasks select nothing.
        """
        if top_k < 1:
            return []
        lowered = task.lower()
        keyword_scores = {
            spec.name: sum(
                1 for keyword in spec.keywords() if keyword in lowered
            )
            for spec in self._skills.values()
        }

        if mode == "keyword" or not self._skills:
            candidates = sorted(
                (
                    (score, spec.name, spec)
                    for spec in self._skills.values()
                    for score in (keyword_scores[spec.name],)
                    if score > 0
                ),
                key=lambda item: (-item[0], item[1]),
            )
            return [spec for _, _, spec in candidates[:top_k]]

        if self._semantic is None:
            self._semantic = SemanticSkillSelector().fit(
                self._skills.values()
            )

        if mode == "semantic":
            best = self._semantic.best(task)
            return [
                self._skills[name]
                for name, _ in best[:top_k]
            ]

        if mode != "hybrid":
            raise ValueError(f"unsupported selection mode: {mode}")

        semantic = self._semantic.score(task)
        ranked = sorted(
            (
                (keyword_scores[name] + 2.0 * semantic.get(name, 0.0), name)
                for name, score in keyword_scores.items()
                if score > 0
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [self._skills[name] for _, name in ranked[:top_k]]
