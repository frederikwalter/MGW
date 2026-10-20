"""The benchmark series (docs/st_3d_alignment_strategies.md §4): embryos, pairs, tiers.

`bench/configs/series.yaml` lists every embryo with its sections (frozen once, so every step uses the same sections)
and every pair. A pair is the unit of work: kind `cross` (human a, mouse b), `between` (two embryos of one species)
or `within` (one embryo; generated as `W:<embryo>` for every embryo of the tier's cross pairs, a leave-one-out over
its own sections).

Directions: a pair has two sides, a and b. Direction `a_stack` takes b's sections as queries and a's as targets,
`b_stack` the reverse. For cross pairs a is human, so a_stack = "human_stack" of the notebook. A within pair has the
one direction `self`.
"""
from dataclasses import dataclass, field
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
    kind: str             # cross | between | within
    a: Embryo
    b: Embryo
    tiers: tuple
    stage: str = ""
    overlap: str = ""
    ceiling: str = ""     # id of the between-embryo ceiling (cross pairs)
    guardrail: str = ""   # "end_of_stack" for the negative control
    note: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def directions(self):
        return ("self",) if self.kind == "within" else ("a_stack", "b_stack")

    def sides(self, direction):
        """(query embryo, target embryo) of a direction."""
        if direction == "self":
            return self.a, self.a
        return (self.b, self.a) if direction == "a_stack" else (self.a, self.b)

    def direction_label(self, direction):
        if self.kind == "cross" and direction != "self":
            return {"a_stack": "human_stack", "b_stack": "mouse_stack"}[direction]
        if direction == "self":
            return "self"
        return f"{self.a.name if direction == 'a_stack' else self.b.name}_stack"

    @property
    def embryos(self):
        return (self.a,) if self.kind == "within" else (self.a, self.b)

    @property
    def ceiling_ids(self):
        """Ceilings of a cross pair: its between-embryo ceiling (if any) and the W of both embryos."""
        out = [self.ceiling] if self.ceiling else []
        return out + [within_id(self.a.name), within_id(self.b.name)]


def within_id(embryo):
    return f"W:{embryo}"


@dataclass(frozen=True)
class Series:
    embryos: dict
    pairs: dict           # id -> Pair, the W entries included
    path: Path | None = None

    def tier(self, tier):
        """Pair ids of a tier, cross pairs first, in file order."""
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
        a, b = embryos[p["a"]], embryos[p["b"]]
        kind = p["kind"]
        if kind == "cross" and not (a.dataset == "HESTA" and b.dataset == "MOSTA"):
            raise ValueError(f"pair {pid}: a cross pair has a = HESTA, b = MOSTA")
        if kind == "between" and a.dataset != b.dataset:
            raise ValueError(f"pair {pid}: a between pair has one species")
        known = {"kind", "a", "b", "tiers", "stage", "overlap", "ceiling", "guardrail", "note"}
        pairs[pid] = Pair(id=pid, kind=kind, a=a, b=b, tiers=tuple(p["tiers"]), stage=str(p.get("stage", "")),
                          overlap=str(p.get("overlap", "")), ceiling=str(p.get("ceiling", "") or ""),
                          guardrail=str(p.get("guardrail", "") or ""), note=str(p.get("note", "")),
                          extra={k: v for k, v in p.items() if k not in known})
    # W: one per embryo of the cross pairs, in the tiers of those pairs (plus `within.tiers` if given)
    w_tiers = {}
    for p in pairs.values():
        if p.kind == "cross":
            for e in (p.a, p.b):
                w_tiers.setdefault(e.name, set()).update(p.tiers)
    extra = set((raw.get("within") or {}).get("tiers", []))
    skip = set((raw.get("within") or {}).get("skip", []))
    for name, tiers in w_tiers.items():
        if name in skip or len(embryos[name].sections) < 2:
            continue
        wid = within_id(name)
        pairs[wid] = Pair(id=wid, kind="within", a=embryos[name], b=embryos[name],
                          tiers=tuple(sorted(tiers | extra)))
    for p in pairs.values():
        if p.ceiling and p.ceiling not in pairs:
            raise ValueError(f"pair {p.id}: ceiling {p.ceiling} is not a pair")
    return Series(embryos=embryos, pairs=pairs, path=path)
