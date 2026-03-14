#!/usr/bin/env python3
"""
Grad Ball Seating Chart Optimizer v2
=====================================
Major improvements over v1:
  - Graph-based greedy cluster seeding for smarter initial placement
  - Multiple move types: swap, relocate, chain-swap, friend-group move
  - Adaptive reheating to escape local optima
  - Cluster bonuses for mutual friend groups
  - Tuned for 550 people / 10+ minute runs

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
  python seating_optimizer_v2.py --input responses.csv --tables tables.csv --output seating_chart.csv
  python seating_optimizer_v2.py --input responses.csv --tables tables.csv --output seating_chart.csv --runs 5
"""

import csv
import math
import random
import argparse
import time
from collections import defaultdict
from copy import deepcopy

# ── Column name configuration ──────────────────────────────────────────────────
COLUMN_NAME        = "name"
COLUMN_PLUS_ONE    = "plus one"
COLUMN_PREFERENCES = "who to be seated with UP TO 6"
COLUMN_AVOIDANCES  = "who to not be seated with "

# ── Scoring weights ────────────────────────────────────────────────────────────
WEIGHT_MUTUAL     = 10      # Both listed each other
WEIGHT_ONE_SIDED  = 3       # Only one listed the other
WEIGHT_PLUS_ONE   = 15      # Plus-one bond (very strong)
PENALTY_AVOIDANCE = 100     # Hard penalty for avoidance violations
CLUSTER_BONUS     = 5       # Extra bonus per edge in a 3+ clique at a table

# ── Annealing parameters ───────────────────────────────────────────────────────
INITIAL_TEMP   = 200.0      # Lower start — dataset is small (~100 people)
COOLING_RATE   = 0.9997     # Faster cooling to match smaller search space
MIN_TEMP       = 0.01
ITERATIONS     = 500_000    # Sufficient for ~100 people
REHEAT_INTERVAL = 50_000    # Reheat every N iterations if stuck
REHEAT_TEMP     = 80.0      # Temperature to reheat to
STALE_THRESHOLD = 20_000    # Iterations without improvement before reheat


def normalize_name(name):
    """Normalize to title case so Alice Smith, alice smith, ALICE SMITH all match."""
    return name.strip().title()


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
            row = {k.strip(): v for k, v in row.items()}
            name = normalize_name(row.get(COLUMN_NAME, ""))
            if not name:
                continue
            attendees.append(name)
            all_names.add(name)

            prefs = [normalize_name(p) for p in parse_names(row.get(COLUMN_PREFERENCES, ""))]
            raw_prefs[name] = prefs
            all_names.update(prefs)

            avoids = [normalize_name(a) for a in parse_names(row.get(COLUMN_AVOIDANCES, ""))]
            raw_avoids[name] = avoids
            all_names.update(avoids)

            plus_one = normalize_name(row.get(COLUMN_PLUS_ONE, ""))
            if plus_one:
                plus_one_map[name] = plus_one
                all_names.add(plus_one)

    # Add plus-ones / preference mentions who didn't fill the form
    form_fillers = set(attendees)
    for p in list(all_names):
        if p not in form_fillers:
            attendees.append(p)

    # Build preference graph — edge_weights[sorted pair] = raw weight sum
    edge_weights = defaultdict(int)
    for name, prefs in raw_prefs.items():
        for other in prefs:
            if other in all_names:
                edge_weights[tuple(sorted([name, other]))] += WEIGHT_ONE_SIDED

    # Plus-one bonds override
    for name, plus_one in plus_one_map.items():
        key = tuple(sorted([name, plus_one]))
        edge_weights[key] = max(edge_weights[key], WEIGHT_PLUS_ONE)

    # Upgrade mutual preferences
    pref_weights = {}
    for key, w in edge_weights.items():
        if w >= 2 * WEIGHT_ONE_SIDED and w < WEIGHT_PLUS_ONE:
            pref_weights[key] = WEIGHT_MUTUAL + WEIGHT_ONE_SIDED  # mutual bonus
        else:
            pref_weights[key] = w

    # Build adjacency list for cluster detection
    adjacency = defaultdict(set)
    for (a, b), w in pref_weights.items():
        if w >= WEIGHT_MUTUAL:  # only strong connections for clustering
            adjacency[a].add(b)
            adjacency[b].add(a)

    avoidance_pairs = set()
    for name, avoids in raw_avoids.items():
        for other in avoids:
            avoidance_pairs.add(frozenset([name, other]))

    return attendees, pref_weights, avoidance_pairs, plus_one_map, adjacency


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


# ── IMPROVED: Graph-based greedy cluster seeding ───────────────────────────────

def find_friend_clusters(adjacency, pref_weights, plus_one_map):
    """
    Build clusters of strongly-connected friends using greedy expansion.
    Returns list of clusters (each a list of names), largest first.
    """
    visited = set()
    clusters = []

    # Start from people with the most mutual connections
    people_by_degree = sorted(adjacency.keys(), key=lambda p: len(adjacency[p]), reverse=True)

    for seed in people_by_degree:
        if seed in visited:
            continue

        cluster = [seed]
        visited.add(seed)

        # Greedily expand: add the neighbor most connected to the existing cluster
        candidates = set(adjacency[seed]) - visited
        while candidates:
            best = None
            best_score = 0
            for c in candidates:
                score = sum(1 for m in cluster if c in adjacency.get(m, set()))
                if score > best_score:
                    best = c
                    best_score = score
            if best is None or best_score == 0:
                break
            cluster.append(best)
            visited.add(best)
            candidates = (candidates | (adjacency[best] - visited)) - {best}

        if len(cluster) >= 2:
            clusters.append(cluster)

    # Also ensure plus-one pairs are captured
    for name, po in plus_one_map.items():
        if name not in visited or po not in visited:
            clusters.append([name, po])
            visited.update([name, po])

    clusters.sort(key=len, reverse=True)
    return clusters


def initial_assignment_clustered(attendees, tables_meta, pref_weights, plus_one_map, adjacency):
    """
    Place friend clusters onto tables greedily, then fill remaining seats randomly.
    Much better starting point than pure random.
    """
    clusters = find_friend_clusters(adjacency, pref_weights, plus_one_map)

    num_tables = len(tables_meta)
    tables_people = [[] for _ in range(num_tables)]
    capacities = [t["capacity"] for t in tables_meta]
    placed = set()

    # Sort tables by capacity descending so big clusters go to big tables
    table_order = sorted(range(num_tables), key=lambda i: capacities[i], reverse=True)

    # Place clusters greedily
    for cluster in clusters:
        # Remove already-placed people from cluster
        unplaced = [p for p in cluster if p not in placed]
        if not unplaced:
            continue

        # Find the table with the best fit (enough room, smallest waste)
        best_table = None
        best_space = float("inf")
        for t_idx in table_order:
            remaining = capacities[t_idx] - len(tables_people[t_idx])
            if remaining >= len(unplaced) and remaining < best_space:
                best_table = t_idx
                best_space = remaining

        if best_table is not None:
            tables_people[best_table].extend(unplaced)
            placed.update(unplaced)
        else:
            # Cluster too big — split it. Place as many as fit at best available table.
            for t_idx in table_order:
                remaining = capacities[t_idx] - len(tables_people[t_idx])
                if remaining > 0:
                    chunk = unplaced[:remaining]
                    tables_people[t_idx].extend(chunk)
                    placed.update(chunk)
                    unplaced = unplaced[remaining:]
                    if not unplaced:
                        break

    # Place everyone else randomly
    unplaced_list = [a for a in attendees if a not in placed]
    random.shuffle(unplaced_list)

    for person in unplaced_list:
        for t_idx in table_order:
            if len(tables_people[t_idx]) < capacities[t_idx]:
                tables_people[t_idx].append(person)
                break
        else:
            # Overflow to largest table (shouldn't happen if capacity is correct)
            tables_people[table_order[0]].append(person)

    return tables_people


# ── Scoring with cluster bonuses ───────────────────────────────────────────────

def table_score(table, pref_weights, avoidance_pairs):
    """Score a single table, including cluster bonuses for 3+ connected groups."""
    total = 0.0
    satisfied_edges = 0
    table_set = set(table)

    for i, a in enumerate(table):
        for b in table[i+1:]:
            key = tuple(sorted([a, b]))
            w = pref_weights.get(key, 0)
            if w > 0:
                total += w
                satisfied_edges += 1
            if frozenset([a, b]) in avoidance_pairs:
                total -= PENALTY_AVOIDANCE

    # Cluster bonus: reward tables where many preference edges are satisfied
    # This encourages keeping friend groups intact rather than spreading them thin
    if satisfied_edges >= 3:
        total += satisfied_edges * CLUSTER_BONUS

    return total


def score_assignment(tables_people, pref_weights, avoidance_pairs):
    return sum(table_score(t, pref_weights, avoidance_pairs) for t in tables_people)


# ── Person-to-table index for fast lookups ─────────────────────────────────────

def build_person_index(tables_people):
    """Returns dict mapping person -> (table_index, position_in_table)."""
    idx = {}
    for t, table in enumerate(tables_people):
        for p, person in enumerate(table):
            idx[person] = (t, p)
    return idx


# ── Move types ─────────────────────────────────────────────────────────────────

def move_swap(tables_people, capacities, person_idx):
    """Classic: swap two people from different tables."""
    non_empty = [i for i, t in enumerate(tables_people) if t]
    if len(non_empty) < 2:
        return None
    t1, t2 = random.sample(non_empty, 2)
    i1 = random.randrange(len(tables_people[t1]))
    i2 = random.randrange(len(tables_people[t2]))
    return ("swap", t1, i1, t2, i2)


def move_relocate(tables_people, capacities, person_idx):
    """Move one person from their table to a different table (if room)."""
    non_empty = [i for i, t in enumerate(tables_people) if t]
    not_full = [i for i in range(len(tables_people)) if len(tables_people[i]) < capacities[i]]
    if not non_empty or not not_full:
        return None
    t_from = random.choice(non_empty)
    if len(tables_people[t_from]) <= 1:
        return None  # don't empty a table
    candidates = [t for t in not_full if t != t_from]
    if not candidates:
        return None
    t_to = random.choice(candidates)
    i_from = random.randrange(len(tables_people[t_from]))
    return ("relocate", t_from, i_from, t_to)


def move_chain_swap(tables_people, capacities, person_idx):
    """3-way circular swap: A->B->C->A across three tables."""
    non_empty = [i for i, t in enumerate(tables_people) if len(t) >= 1]
    if len(non_empty) < 3:
        return None
    t1, t2, t3 = random.sample(non_empty, 3)
    i1 = random.randrange(len(tables_people[t1]))
    i2 = random.randrange(len(tables_people[t2]))
    i3 = random.randrange(len(tables_people[t3]))
    return ("chain", t1, i1, t2, i2, t3, i3)


def move_friend_group(tables_people, capacities, person_idx, pref_weights):
    """Move a person AND one of their preferred friends together to a new table."""
    # Pick a random person who has preferences
    all_people = list(person_idx.keys())
    random.shuffle(all_people)

    for person in all_people[:20]:  # try up to 20 people
        t_p, i_p = person_idx[person]
        # Find a friend at the same table
        friends_here = []
        for other in tables_people[t_p]:
            if other != person:
                key = tuple(sorted([person, other]))
                if key in pref_weights and pref_weights[key] >= WEIGHT_MUTUAL:
                    friends_here.append(other)
        if not friends_here:
            continue

        friend = random.choice(friends_here)
        t_f, i_f = person_idx[friend]
        assert t_f == t_p

        # Find a destination table with room for 2
        not_full = [i for i in range(len(tables_people))
                    if i != t_p and (capacities[i] - len(tables_people[i])) >= 2]
        if not not_full:
            # Try swapping pair with 2 people from another table
            other_tables = [i for i in range(len(tables_people))
                           if i != t_p and len(tables_people[i]) >= 2]
            if not other_tables:
                continue
            t_dest = random.choice(other_tables)
            swap_indices = random.sample(range(len(tables_people[t_dest])), 2)
            return ("friend_swap", t_p, person, friend, t_dest, swap_indices[0], swap_indices[1])

        t_dest = random.choice(not_full)
        return ("friend_move", t_p, person, friend, t_dest)

    return None


def apply_move(move, tables_people, person_idx):
    """Apply a move and return info needed to compute delta and undo."""
    kind = move[0]
    affected_tables = set()

    if kind == "swap":
        _, t1, i1, t2, i2 = move
        tables_people[t1][i1], tables_people[t2][i2] = tables_people[t2][i2], tables_people[t1][i1]
        # Update index
        p1, p2 = tables_people[t1][i1], tables_people[t2][i2]
        person_idx[p1] = (t1, i1)
        person_idx[p2] = (t2, i2)
        affected_tables = {t1, t2}

    elif kind == "relocate":
        _, t_from, i_from, t_to = move
        person = tables_people[t_from].pop(i_from)
        tables_people[t_to].append(person)
        # Rebuild index for affected tables
        for idx_pos, p in enumerate(tables_people[t_from]):
            person_idx[p] = (t_from, idx_pos)
        person_idx[person] = (t_to, len(tables_people[t_to]) - 1)
        affected_tables = {t_from, t_to}

    elif kind == "chain":
        _, t1, i1, t2, i2, t3, i3 = move
        # Circular: t1->t2, t2->t3, t3->t1
        p1 = tables_people[t1][i1]
        p2 = tables_people[t2][i2]
        p3 = tables_people[t3][i3]
        tables_people[t1][i1] = p3
        tables_people[t2][i2] = p1
        tables_people[t3][i3] = p2
        person_idx[p1] = (t2, i2)
        person_idx[p2] = (t3, i3)
        person_idx[p3] = (t1, i1)
        affected_tables = {t1, t2, t3}

    elif kind == "friend_move":
        _, t_src, person, friend, t_dest = move
        # Remove both from source
        tables_people[t_src].remove(person)
        tables_people[t_src].remove(friend)
        tables_people[t_dest].append(person)
        tables_people[t_dest].append(friend)
        # Rebuild source index
        for idx_pos, p in enumerate(tables_people[t_src]):
            person_idx[p] = (t_src, idx_pos)
        person_idx[person] = (t_dest, len(tables_people[t_dest]) - 2)
        person_idx[friend] = (t_dest, len(tables_people[t_dest]) - 1)
        affected_tables = {t_src, t_dest}

    elif kind == "friend_swap":
        _, t_src, person, friend, t_dest, di1, di2 = move
        # Swap the pair with two people at destination
        dest_p1 = tables_people[t_dest][di1]
        dest_p2 = tables_people[t_dest][di2]
        # Remove pair from source, add dest pair
        tables_people[t_src].remove(person)
        tables_people[t_src].remove(friend)
        tables_people[t_src].extend([dest_p1, dest_p2])
        # Remove dest pair, add source pair
        # Remove by value (careful with indices shifting)
        tables_people[t_dest] = [p for p in tables_people[t_dest] if p not in (dest_p1, dest_p2)]
        tables_people[t_dest].extend([person, friend])
        # Rebuild indices for both tables
        for idx_pos, p in enumerate(tables_people[t_src]):
            person_idx[p] = (t_src, idx_pos)
        for idx_pos, p in enumerate(tables_people[t_dest]):
            person_idx[p] = (t_dest, idx_pos)
        affected_tables = {t_src, t_dest}

    return affected_tables


def undo_move(move, tables_people, person_idx):
    """Undo a previously applied move."""
    kind = move[0]

    if kind == "swap":
        _, t1, i1, t2, i2 = move
        # Just swap back
        tables_people[t1][i1], tables_people[t2][i2] = tables_people[t2][i2], tables_people[t1][i1]
        p1, p2 = tables_people[t1][i1], tables_people[t2][i2]
        person_idx[p1] = (t1, i1)
        person_idx[p2] = (t2, i2)

    elif kind == "relocate":
        _, t_from, i_from, t_to = move
        person = tables_people[t_to].pop()
        tables_people[t_from].insert(i_from, person)
        for idx_pos, p in enumerate(tables_people[t_from]):
            person_idx[p] = (t_from, idx_pos)
        for idx_pos, p in enumerate(tables_people[t_to]):
            person_idx[p] = (t_to, idx_pos)

    elif kind == "chain":
        _, t1, i1, t2, i2, t3, i3 = move
        # Reverse circular
        p3 = tables_people[t1][i1]
        p1 = tables_people[t2][i2]
        p2 = tables_people[t3][i3]
        tables_people[t1][i1] = p1
        tables_people[t2][i2] = p2
        tables_people[t3][i3] = p3
        person_idx[p1] = (t1, i1)
        person_idx[p2] = (t2, i2)
        person_idx[p3] = (t3, i3)

    elif kind == "friend_move":
        _, t_src, person, friend, t_dest = move
        tables_people[t_dest].remove(person)
        tables_people[t_dest].remove(friend)
        tables_people[t_src].append(person)
        tables_people[t_src].append(friend)
        for idx_pos, p in enumerate(tables_people[t_src]):
            person_idx[p] = (t_src, idx_pos)
        for idx_pos, p in enumerate(tables_people[t_dest]):
            person_idx[p] = (t_dest, idx_pos)

    elif kind == "friend_swap":
        _, t_src, person, friend, t_dest, di1, di2 = move
        # The pair is now at t_dest, the swapped people at t_src — reverse it
        dest_p1_candidates = [p for p in tables_people[t_src] if p not in
                              {pp for pp in tables_people[t_src]} - {tables_people[t_src][-1], tables_people[t_src][-2]}]
        # Simpler: just recompute by applying the reverse swap
        # Remove pair from dest, the swapped pair from src
        tables_people[t_dest].remove(person)
        tables_people[t_dest].remove(friend)
        # The last two added to t_src were dest_p1, dest_p2
        dest_p1 = tables_people[t_src].pop()
        dest_p2 = tables_people[t_src].pop()
        # Put them back
        tables_people[t_dest].insert(di1, dest_p1)
        tables_people[t_dest].insert(di2, dest_p2)
        tables_people[t_src].append(person)
        tables_people[t_src].append(friend)
        # Note: this undo is imperfect for ordering but the scores don't depend on order
        for idx_pos, p in enumerate(tables_people[t_src]):
            person_idx[p] = (t_src, idx_pos)
        for idx_pos, p in enumerate(tables_people[t_dest]):
            person_idx[p] = (t_dest, idx_pos)


# ── Main annealing loop ───────────────────────────────────────────────────────

def run_annealing(attendees, pref_weights, avoidance_pairs, plus_one_map, adjacency,
                  tables_meta, seed=None):
    if seed is not None:
        random.seed(seed)

    capacities = [t["capacity"] for t in tables_meta]

    # Smart initial placement
    tables_people = initial_assignment_clustered(
        attendees, tables_meta, pref_weights, plus_one_map, adjacency
    )
    person_idx = build_person_index(tables_people)

    # Pre-compute table scores
    t_scores = [table_score(t, pref_weights, avoidance_pairs) for t in tables_people]
    current_score = sum(t_scores)
    best_tables   = deepcopy(tables_people)
    best_score    = current_score
    temp          = INITIAL_TEMP

    last_improvement = 0
    move_type_weights = [50, 20, 15, 15]  # swap, relocate, chain, friend_group

    start_time = time.time()

    for iteration in range(ITERATIONS):
        if temp < MIN_TEMP:
            temp = MIN_TEMP  # floor but don't stop

        # Adaptive reheating
        if iteration - last_improvement > STALE_THRESHOLD and iteration % REHEAT_INTERVAL == 0:
            temp = REHEAT_TEMP
            # Shift move weights toward more exploratory moves
            move_type_weights = [35, 25, 20, 20]

        # Choose move type
        r = random.random() * sum(move_type_weights)
        cumulative = 0
        move = None
        for idx, w in enumerate(move_type_weights):
            cumulative += w
            if r <= cumulative:
                if idx == 0:
                    move = move_swap(tables_people, capacities, person_idx)
                elif idx == 1:
                    move = move_relocate(tables_people, capacities, person_idx)
                elif idx == 2:
                    move = move_chain_swap(tables_people, capacities, person_idx)
                elif idx == 3:
                    move = move_friend_group(tables_people, capacities, person_idx, pref_weights)
                break

        if move is None:
            move = move_swap(tables_people, capacities, person_idx)
        if move is None:
            continue

        # Compute scores of affected tables BEFORE
        kind = move[0]
        if kind == "swap":
            affected = {move[1], move[3]}
        elif kind == "relocate":
            affected = {move[1], move[3]}
        elif kind == "chain":
            affected = {move[1], move[3], move[5]}
        elif kind in ("friend_move", "friend_swap"):
            affected = {move[1], move[4]}
        else:
            affected = set()

        score_before = sum(t_scores[t] for t in affected)

        # Apply move
        actually_affected = apply_move(move, tables_people, person_idx)

        # Compute scores AFTER
        new_t_scores = {}
        for t in actually_affected:
            new_t_scores[t] = table_score(tables_people[t], pref_weights, avoidance_pairs)
        score_after = sum(new_t_scores.get(t, t_scores[t]) for t in affected)

        delta = score_after - score_before

        if delta >= 0 or random.random() < math.exp(delta / temp):
            # Accept
            for t, s in new_t_scores.items():
                t_scores[t] = s
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_tables = deepcopy(tables_people)
                last_improvement = iteration
        else:
            # Reject — undo
            undo_move(move, tables_people, person_idx)

        temp *= COOLING_RATE

        # Progress reporting
        if iteration % 250_000 == 0 and iteration > 0:
            elapsed = time.time() - start_time
            print(f"    iter {iteration:>10,}  score={current_score:>8.1f}  "
                  f"best={best_score:>8.1f}  temp={temp:.2f}  "
                  f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - start_time
    print(f"    Final: score={best_score:.1f}  elapsed={elapsed:.0f}s")
    return best_tables, best_score


def write_simple_output(path, tables_people, tables_meta):
    """Write a plain text file listing each table and its occupants."""
    with open(path, "w", encoding="utf-8") as f:
        for meta, table in zip(tables_meta, tables_people):
            names = ", ".join(sorted(table))
            f.write(f"{meta['name']}: {names}\n")


def write_output(path, tables_people, tables_meta, pref_weights, avoidance_pairs, plus_one_map):
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
    fulfilled    = sum(1 for (a, b) in pref_weights if person_table.get(a) == person_table.get(b))
    violations   = sum(1 for pair in avoidance_pairs
                       if len(pair) == 2 and person_table.get(list(pair)[0]) == person_table.get(list(pair)[1]))
    po_total     = len(plus_one_map)
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
    print(f"  Plus-ones together:     {po_total - po_separated} / {po_total}")
    print(f"  Plus-ones separated:    {po_separated}  ← should be 0")
    print(f"  Output written to:      {path}")
    print(f"{'='*52}\n")


def main():
    global ITERATIONS
    parser = argparse.ArgumentParser(description="Grad Ball Seating Optimizer v2")
    parser.add_argument("--input",      required=True, help="Google Form responses CSV")
    parser.add_argument("--tables",     required=True, help="Table names & capacities CSV")
    parser.add_argument("--output",     required=True, help="Output seating chart CSV")
    parser.add_argument("--runs",       type=int, default=3,
                        help="Independent runs — best kept (default: 3)")
    parser.add_argument("--simple-output", default=None,
                        help="Optional path for plain-text table summary")
    parser.add_argument("--iterations", type=int, default=1_500_000,
                        help="Iterations per run (default: 1500000)")
    args       = parser.parse_args()
    ITERATIONS = args.iterations

    print(f"\n{'='*52}")
    print(f"  Grad Ball Seating Optimizer v2")
    print(f"{'='*52}")

    print(f"\nLoading tables from:    {args.tables}")
    tables_meta = load_tables(args.tables)
    if not tables_meta:
        raise ValueError("No tables loaded — check your tables.csv format.")
    for t in tables_meta:
        print(f"  {t['name']:<30} capacity {t['capacity']}")

    print(f"\nLoading attendees from: {args.input}")
    attendees, pref_weights, avoidance_pairs, plus_one_map, adjacency = load_attendees(args.input)
    print(f"  {len(pref_weights)} preference pairs")
    print(f"  {len(avoidance_pairs)} avoidance constraints")
    print(f"  {len(plus_one_map)} plus-one pairs")

    mutual = sum(1 for w in pref_weights.values() if w >= WEIGHT_MUTUAL)
    print(f"  {mutual} mutual preference pairs (both listed each other)")

    print()
    validate_capacity(attendees, tables_meta)

    best_tables = None
    best_score  = float("-inf")

    print()
    for run in range(1, args.runs + 1):
        print(f"Run {run}/{args.runs}:")
        tables_people, score = run_annealing(
            attendees, pref_weights, avoidance_pairs, plus_one_map, adjacency,
            tables_meta, seed=run
        )
        print(f"  → score = {score:.1f}")
        if score > best_score:
            best_score  = score
            best_tables = tables_people

    print(f"\nBest score across all runs: {best_score:.1f}")
    write_output(args.output, best_tables, tables_meta, pref_weights, avoidance_pairs, plus_one_map)
    if args.simple_output:
        write_simple_output(args.simple_output, best_tables, tables_meta)
        print(f"Simple table list written to: {args.simple_output}")


if __name__ == "__main__":
    main()