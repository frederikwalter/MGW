"""The benchmark series: embryos, cross-species pairs, tiers.

`bench/configs/series.yaml` lists every embryo with its sections (frozen once, so every run uses the same sections)
and every pair. A pair (human a, mouse b) is the unit of work.

Directions: direction `a_stack` takes b's sections as queries and a's as targets (the human stack), `b_stack` the
reverse (the mouse stack).
"""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Embryo:
    name: str
    dataset: str          # HESTA | MOSTA
    stage: str
    sections: tuple

    @property
    def species(self):
        return "human" if self.dataset == "HESTA" else "mouse"


@dataclass(frozen=True)
class Pair:
    id: str
    a: Embryo
    b: Embryo
    tiers: tuple
    stage: str = ""
    overlap: str = ""
    note: str = ""
    kind: str = "cross"

    directions = ("a_stack", "b_stack")

    def sides(self, direction):
        """(query embryo, target embryo) of a direction."""
        return (self.b, self.a) if direction == "a_stack" else (self.a, self.b)

    def direction_label(self, direction):
        return {"a_stack": "human_stack", "b_stack": "mouse_stack"}[direction]

    @property
    def embryos(self):
        return (self.a, self.b)


@dataclass(frozen=True)
class Series:
    embryos: dict
    pairs: dict           # id -> Pair
    path: Path | None = None

    def tier(self, tier):
        """Pair ids of a tier, in file order."""
        return [p for p in self.pairs if tier in self.pairs[p].tiers]

    def all_sections(self, tier=None):
        """(dataset, section) of every section used by the tier's pairs, without duplicates."""
        seen, out = set(), []
        for pid in (self.tier(tier) if tier else self.pairs):
            for e in self.pairs[pid].embryos:
                for s in e.sections:
                    if s not in seen:
                        seen.add(s)
                        out.append((e.dataset, s))
        return out


def load(path):
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    embryos = {}
    for name, e in raw["embryos"].items():
        if e["dataset"] not in ("HESTA", "MOSTA"):
            raise ValueError(f"embryo {name}: dataset must be HESTA or MOSTA")
        embryos[name] = Embryo(name=name, dataset=e["dataset"], stage=str(e.get("stage", "")),
                               sections=tuple(e["sections"]))
    pairs = {}
    for pid, p in raw["pairs"].items():
        unknown = set(p) - {"a", "b", "tiers", "stage", "overlap", "note"}
        if unknown:
            raise KeyError(f"pair {pid}: unknown keys {sorted(unknown)}")
        a, b = embryos[p["a"]], embryos[p["b"]]
        if not (a.dataset == "HESTA" and b.dataset == "MOSTA"):
            raise ValueError(f"pair {pid}: a = HESTA, b = MOSTA")
        pairs[pid] = Pair(id=pid, a=a, b=b, tiers=tuple(p["tiers"]), stage=str(p.get("stage", "")),
                          overlap=str(p.get("overlap", "")), note=str(p.get("note", "")))
    return Series(embryos=embryos, pairs=pairs, path=path)
