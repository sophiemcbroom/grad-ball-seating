#!/usr/bin/env python3
"""
Grad Ball Seating Chart Optimizer
==================================
Uses simulated annealing to assign attendees to named tables with specific capacities,
maximizing preference satisfaction while enforcing hard "do not seat with" constraints.

REQUIRED FILES:
  1. responses.csv   — Google Form export (one row per attendee)
  2. tables.csv      — Your venue's table list (one row per table)

tables.csv format — two columns, header optional:
  Table Name, Capacity
  Table 1, 14
  Table 2, 12
  Professors Table, 11
  Round Table A, 16

responses.csv expected columns (adjust COLUMN_* constants below if names differ):
  - Name
  - Do you have a plus one? If so, who?
  - Who would you like to be seated with?
  - Is there anyone you would prefer not to be seated with...

Usage:
  python seating_optimizer.py --input responses.csv --tables tables.csv --output seating_chart.csv
  python seating_optimizer.py --input responses.csv --tables tables.csv --output seating_chart.csv --runs 5
"""

import csv
import math
import random
import argparse
from collections import defaultdict
from copy import deepcopy

# ── Column name configuration ──────────────────────────────────────────────────
COLUMN_NAME        = "Name"
COLUMN_PLUS_ONE    = "Do you have a plus one? If so, who?"
COLUMN_PREFERENCES = "Who would you like to be seated with?"
COLUMN_AVOIDANCES  = "Is there anyone you would prefer not to be seated with due to past experiences"

# ── Scoring weights ────────────────────────────────────────────────────────────
WEIGHT_MUTUAL     = 3
WEIGHT_ONE_SIDED  = 1
WEIGHT_PLUS_ONE   = 10
PENALTY_AVOIDANCE = 50

# ── Annealing parameters ───────────────────────────────────────────────────────
INITIAL_TEMP = 500.0
COOLING_RATE = 0.9995
MIN_TEMP     = 0.1
ITERATIONS   = 200_000


def parse_names(raw):
    if not raw or not raw.strip():
        return []
    return [n.strip() for n in raw.split(",") if n.strip()]


def load_tables(path):
    tables = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            name_cell = row[0].strip()
            cap_cell  = row[1].strip()
            if i == 0 and not cap_cell.isdigit():
                continue  # skip header
            try:
                capacity = int(cap_cell)
            except ValueError:
                print(f"  Warning: skipping unreadable row: {row}")
                continue
            tables.append({"name": name_cell, "capacity": capacity})
    return tables


def load_attendees(path):
    attendees    = []
    raw_prefs    = {}
    raw_avoids   = {}
    plus_one_map = {}
    all_names    = set()

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get(COLUMN_NAME, "").strip()
            if not name:
                continue
            attendees.append(name)
            all_names.add(name)

            prefs = parse_names(row.get(COLUMN_PREFERENCES, ""))
            raw_prefs[name] = prefs
            all_names.update(prefs)

            avoids = parse_names(row.get(COLUMN_AVOIDANCES, ""))
            raw_avoids[name] = avoids
            all_names.update(avoids)

            plus_one = row.get(COLUMN_PLUS_ONE, "").strip()
            if plus_one:
                plus_one_map[name] = plus_one
                all_names.add(plus_one)

    # Add plus-ones who didn't fill the form
    form_fillers = set(attendees)
    for p in list(all_names):
        if p not in form_fillers:
            attendees.append(p)

    # Build preference graph
    edge_weights = defaultdict(int)
    for name, prefs in raw_prefs.items():
        for other in prefs:
            if other in all_names:
                edge_weights[tuple(sorted([name, other]))] += WEIGHT_ONE_SIDED

    for name, plus_one in plus_one_map.items():
        key = tuple(sorted([name, plus_one]))
        edge_weights[key] = max(edge_weights[key], WEIGHT_PLUS_ONE)

    pref_weights = {}
    for key, w in edge_weights.items():
        pref_weights[key] = (WEIGHT_MUTUAL + WEIGHT_ONE_SIDED) if w >= 2 * WEIGHT_ONE_SIDED else w

    avoidance_pairs = set()
    for name, avoids in raw_avoids.items():
        for other in avoids:
            avoidance_pairs.add(frozenset([name, other]))

    return attendees, pref_weights, avoidance_pairs, plus_one_map


def validate_capacity(attendees, tables):
    n         = len(attendees)
    total_cap = sum(t["capacity"] for t in tables)
    slack     = total_cap - n
    print(f"  Attendees:      {n}")
    print(f"  Tables:         {len(tables)}")
    print(f"  Total capacity: {total_cap}  (slack: {slack:+d} seats)")
    if slack < 0:
        raise ValueError(
            f"Not enough seats! {n} attendees but only {total_cap} seats. "
            "Add tables or increase capacities in tables.csv."
        )
    if slack > total_cap * 0.2:
        print(f"  Warning: {slack} empty seats — tables will be noticeably underfilled.")


def initial_assignment(attendees, tables, plus_one_map):
    name_set = set(attendees)
    paired   = set()
    ordered  = []

    for name, plus_one in plus_one_map.items():
        if name not in paired and plus_one in name_set and plus_one not in paired:
            ordered.extend([name, plus_one])
            paired.update([name, plus_one])

    unpaired = [a for a in attendees if a not in paired]
    random.shuffle(unpaired)
    ordered.extend(unpaired)

    result = []
    idx = 0
    for t in tables:
        cap = t["capacity"]
        result.append(ordered[idx:idx + cap])
        idx += cap
    if idx < len(ordered):
        result[-1].extend(ordered[idx:])
    return result


def table_score(table, pref_weights, avoidance_pairs):
    total = 0.0
    for i, a in enumerate(table):
        for b in table[i+1:]:
            key = tuple(sorted([a, b]))
            total += pref_weights.get(key, 0)
            if frozenset([a, b]) in avoidance_pairs:
                total -= PENALTY_AVOIDANCE
    return total


def score_assignment(tables_people, pref_weights, avoidance_pairs):
    return sum(table_score(t, pref_weights, avoidance_pairs) for t in tables_people)


def run_annealing(attendees, pref_weights, avoidance_pairs, plus_one_map, tables_meta, seed=None):
    if seed is not None:
        random.seed(seed)

    tables_people = initial_assignment(attendees, tables_meta, plus_one_map)
    current_score = score_assignment(tables_people, pref_weights, avoidance_pairs)
    best_tables   = deepcopy(tables_people)
    best_score    = current_score
    temp          = INITIAL_TEMP

    for iteration in range(ITERATIONS):
        if temp < MIN_TEMP:
            break

        # Pick two different non-empty tables
        non_empty = [i for i, t in enumerate(tables_people) if t]
        if len(non_empty) < 2:
            break
        t1, t2 = random.sample(non_empty, 2)
        i1 = random.randrange(len(tables_people[t1]))
        i2 = random.randrange(len(tables_people[t2]))

        score_before = (table_score(tables_people[t1], pref_weights, avoidance_pairs) +
                        table_score(tables_people[t2], pref_weights, avoidance_pairs))

        tables_people[t1][i1], tables_people[t2][i2] = tables_people[t2][i2], tables_people[t1][i1]

        score_after = (table_score(tables_people[t1], pref_weights, avoidance_pairs) +
                       table_score(tables_people[t2], pref_weights, avoidance_pairs))

        delta = score_after - score_before

        if delta >= 0 or random.random() < math.exp(delta / temp):
            current_score += delta
            if current_score > best_score:
                best_score  = current_score
                best_tables = deepcopy(tables_people)
        else:
            tables_people[t1][i1], tables_people[t2][i2] = tables_people[t2][i2], tables_people[t1][i1]

        temp *= COOLING_RATE

    return best_tables, best_score


def write_output(path, tables_people, tables_meta, pref_weights, avoidance_pairs, plus_one_map):
    # Build person -> table index map
    person_table = {}
    for t_idx, table in enumerate(tables_people):
        for person in table:
            person_table[person] = t_idx

    # Satisfaction stats
    stats = {}
    for (a, b) in pref_weights:
        same = person_table.get(a) == person_table.get(b)
        for x, y in [(a, b), (b, a)]:
            stats.setdefault(x, {"seated_with": [], "missed": [], "violations": []})
            (stats[x]["seated_with"] if same else stats[x]["missed"]).append(y)

    for pair in avoidance_pairs:
        pl = list(pair)
        if len(pl) == 2:
            a, b = pl
            if person_table.get(a) == person_table.get(b):
                for x, y in [(a, b), (b, a)]:
                    stats.setdefault(x, {"seated_with": [], "missed": [], "violations": []})
                    stats[x]["violations"].append(y)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Table Name", "Capacity", "Filled", "Name",
            "Preferences Fulfilled", "Preferences Missed",
            "Avoidance Violations", "Plus-One Together?"
        ])
        for meta, table in zip(tables_meta, tables_people):
            for person in sorted(table):
                s        = stats.get(person, {"seated_with": [], "missed": [], "violations": []})
                plus_one = plus_one_map.get(person, "")
                po_status = ""
                if plus_one:
                    po_status = "✓" if person_table.get(person) == person_table.get(plus_one) else "✗ SEPARATED"
                writer.writerow([
                    meta["name"], meta["capacity"], len(table), person,
                    "; ".join(s["seated_with"]),
                    "; ".join(s["missed"]),
                    "; ".join(s["violations"]) if s["violations"] else "",
                    po_status,
                ])

    # Summary
    total_edges  = len(pref_weights)
    fulfilled    = sum(1 for (a,b) in pref_weights if person_table.get(a) == person_table.get(b))
    violations   = sum(1 for pair in avoidance_pairs
                       if len(pair)==2 and person_table.get(list(pair)[0])==person_table.get(list(pair)[1]))
    po_separated = sum(1 for name, po in plus_one_map.items()
                       if person_table.get(name) != person_table.get(po))
    total_seated = sum(len(t) for t in tables_people)
    total_cap    = sum(t["capacity"] for t in tables_meta)

    print(f"\n{'='*52}")
    print(f"  SEATING SUMMARY")
    print(f"{'='*52}")
    print(f"  Attendees seated:       {total_seated} / {total_cap} seats used")
    print(f"  Tables:                 {len(tables_meta)}")
    print(f"  Preference pairs:       {total_edges}")
    print(f"  Preferences fulfilled:  {fulfilled} / {total_edges}  ({100*fulfilled//max(total_edges,1)}%)")
    print(f"  Avoidance violations:   {violations}  ← should be 0")
    print(f"  Plus-ones separated:    {po_separated}  ← should be 0")
    print(f"  Output written to:      {path}")
    print(f"{'='*52}\n")


def main():
    global ITERATIONS
    parser = argparse.ArgumentParser(description="Grad Ball Seating Optimizer")
    parser.add_argument("--input",      required=True, help="Google Form responses CSV")
    parser.add_argument("--tables",     required=True, help="Table names & capacities CSV")
    parser.add_argument("--output",     required=True, help="Output seating chart CSV")
    parser.add_argument("--runs",       type=int, default=3,
                        help="Independent runs — best kept (default: 3)")
    parser.add_argument("--iterations", type=int, default=200_000,
                        help="Iterations per run (default: 200000; try 500000 for better results)")
    args       = parser.parse_args()
    ITERATIONS = args.iterations

    print(f"\nLoading tables from:    {args.tables}")
    tables_meta = load_tables(args.tables)
    if not tables_meta:
        raise ValueError("No tables loaded — check your tables.csv format.")
    for t in tables_meta:
        print(f"  {t['name']:<30} capacity {t['capacity']}")

    print(f"\nLoading attendees from: {args.input}")
    attendees, pref_weights, avoidance_pairs, plus_one_map = load_attendees(args.input)
    print(f"  {len(pref_weights)} preference pairs")
    print(f"  {len(avoidance_pairs)} avoidance constraints")
    print(f"  {len(plus_one_map)} plus-one pairs")

    print()
    validate_capacity(attendees, tables_meta)

    best_tables = None
    best_score  = float("-inf")

    print()
    for run in range(1, args.runs + 1):
        print(f"Run {run}/{args.runs} ...", end=" ", flush=True)
        tables_people, score = run_annealing(
            attendees, pref_weights, avoidance_pairs, plus_one_map, tables_meta, seed=run
        )
        print(f"score = {score:.1f}")
        if score > best_score:
            best_score  = score
            best_tables = tables_people

    print(f"\nBest score across all runs: {best_score:.1f}")
    write_output(args.output, best_tables, tables_meta, pref_weights, avoidance_pairs, plus_one_map)


if __name__ == "__main__":
    main()