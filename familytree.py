"""
PRISHA GUPTA (UNDERGRAD)
--------------------------
CS 562 — Assignment 1: Kids Running in the Yard
Family tree generator

Place this file in the same directory as:
    life_expectancy.csv         — columns: Year, Period life expectancy at birth
    first_names.csv             — columns: decade, gender, name, frequency
    last_names.csv              — columns: name, rank  (rank 1–30)
    rank_to_probability.csv     — single row of 30 comma-separated probabilities
    birth_and_marriage_rates.csv — columns: decade, birth_rate, marriage_rate
    gender_name_probability.csv — columns: decade, gender, probability  (optional)
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Person
# ---------------------------------------------------------------------------

@dataclass
class Person:
    """Represents one individual in the family tree."""

    first_name: str
    last_name: str
    year_born: int
    year_died: int
    gender: str                          # "male" or "female"
    partner: Optional["Person"] = None
    children: List["Person"] = field(default_factory=list)

    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    def has_partner(self) -> bool:
        return self.partner is not None

    def set_partner(self, other: "Person") -> None:
        self.partner = other

    def add_child(self, child: "Person") -> None:
        self.children.append(child)


# ---------------------------------------------------------------------------
# PersonFactory: reads data files and creates Person instances
# ---------------------------------------------------------------------------

class PersonFactory:
    """Reads CSV files and generates Person objects."""

    def __init__(self) -> None:
        # decade (int) -> life expectancy (float)  e.g. 1950 -> 62.something
        self._life_expectancy: Dict[int, float] = {}

        # decade (int) -> birth_rate (float)
        self._birth_rate: Dict[int, float] = {}

        # decade (int) -> marriage_rate (float)
        self._marriage_rate: Dict[int, float] = {}

        # (decade_str, gender_str) -> (names, weights)  e.g. ("1950s","male") -> (["James",...],[0.14,...])
        self._first_names: Dict[Tuple[str, str], Tuple[List[str], List[float]]] = {}

        # parallel lists: last name string and its probability weight
        self._last_name_list: List[str] = []
        self._last_name_weights: List[float] = []

        # decade_str -> {"male": prob, "female": prob}
        self._gender_prob: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public: read all files
    # ------------------------------------------------------------------

    def read_files(self) -> None:
        self._read_life_expectancy("life_expectancy.csv")
        self._read_birth_and_marriage("birth_and_marriage_rates.csv")
        self._read_first_names("first_names.csv")
        self._read_last_names("last_names.csv", "rank_to_probability.csv")
        self._read_gender_probability("gender_name_probability.csv")

    # ------------------------------------------------------------------
    # Public: create people
    # ------------------------------------------------------------------

    def get_person(
        self,
        year_born: int,
        last_name: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> Person:
        """Create and return a new Person born in year_born."""
        if gender is None:
            gender = self._sample_gender(year_born)
        first_name = self._sample_first_name(year_born, gender)
        if last_name is None:
            last_name = self._sample_last_name()
        year_died = self._sample_year_died(year_born)
        return Person(
            first_name=first_name,
            last_name=last_name,
            year_born=year_born,
            year_died=year_died,
            gender=gender,
        )

    def maybe_create_partner(self, person: Person) -> Optional[Person]:
        """Return a partner for person based on marriage rate, or None."""
        decade_str = _decade_str(person.year_born)
        marriage_rate = self._marriage_rate.get(_decade_int(person.year_born), 0.0)

        if random.random() > marriage_rate:
            return None

        partner_year = max(1900, min(2120, person.year_born + random.randint(-10, 10)))
        partner = self.get_person(partner_year)
        return partner

    def create_children(self, parent: Person) -> List[Person]:
        """Return a list of children for parent, using birth rates."""
        decade_int = _decade_int(parent.year_born)
        birth_rate = self._birth_rate.get(decade_int, 0.0)

        low = math.ceil(max(0.0, birth_rate - 1.5))
        high = math.ceil(max(0.0, birth_rate + 1.5))
        if high < low:
            high = low

        n_children = random.randint(int(low), int(high))

        # CS 562: a person without a partner has 1 fewer child
        if not parent.has_partner():
            n_children = max(0, n_children - 1)

        if n_children == 0:
            return []

        start_year = parent.year_born + 25
        end_year = parent.year_born + 45
        birth_years = _spread_years(start_year, end_year, n_children)

        children: List[Person] = []
        for year in birth_years:
            if year > 2120:
                continue
            child = self.get_person(year_born=year, last_name=parent.last_name)
            children.append(child)
        return children

    # ------------------------------------------------------------------
    # Private: CSV readers
    # ------------------------------------------------------------------

    def _read_life_expectancy(self, filename: str) -> None:
        """
        File format:
            Year,Period life expectancy at birth
            1950,61.95
            ...
        We average by decade.
        """
        decade_totals: Dict[int, List[float]] = {}
        with open(filename, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                year_val = _safe_int(list(row.values())[0])
                exp_val = _safe_float(list(row.values())[1])
                if year_val is None or exp_val is None:
                    continue
                dec = _decade_int(year_val)
                decade_totals.setdefault(dec, []).append(exp_val)

        for dec, vals in decade_totals.items():
            self._life_expectancy[dec] = sum(vals) / len(vals)

    def _read_birth_and_marriage(self, filename: str) -> None:
        """
        File format:
            decade,birth_rate,marriage_rate
            1950s,3.24,0.78
            ...
        """
        with open(filename, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dec = _parse_decade_str(row["decade"])
                if dec is None:
                    continue
                birth = _safe_float(row["birth_rate"])
                marriage = _safe_float(row["marriage_rate"])
                if birth is not None:
                    self._birth_rate[dec] = birth
                if marriage is not None:
                    self._marriage_rate[dec] = marriage

    def _read_first_names(self, filename: str) -> None:
        """
        File format:
            decade,gender,name,frequency
            1950s,male,James,0.139496
            ...
        """
        tmp: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
        with open(filename, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                decade = row["decade"].strip()
                gender = row["gender"].strip().lower()
                name = row["name"].strip()
                freq = _safe_float(row["frequency"])
                if not name or freq is None:
                    continue
                tmp.setdefault((decade, gender), []).append((name, freq))

        for key, pairs in tmp.items():
            names = [p[0] for p in pairs]
            weights = [p[1] for p in pairs]
            self._first_names[key] = (names, weights)

    def _read_last_names(self, last_names_file: str, rank_prob_file: str) -> None:
        """
        rank_to_probability.csv: single row of 30 comma-separated probabilities.
            Probability for rank 1 = first value, rank 2 = second, etc.

        last_names.csv: expected columns: name, rank
        """
        # Read rank -> probability
        rank_to_prob: Dict[int, float] = {}
        with open(rank_prob_file, newline="", encoding="utf-8-sig") as f:
            content = f.read().strip()
        values = [v.strip() for v in content.split(",") if v.strip()]
        for i, val in enumerate(values):
            prob = _safe_float(val)
            if prob is not None:
                rank_to_prob[i + 1] = prob  # rank 1-indexed

        # Read last names + rank
        with open(last_names_file, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try common column name variants
                name = (row.get("name") or row.get("Name") or row.get("last_name") or "").strip()
                rank_str = (row.get("rank") or row.get("Rank") or row.get("r") or "").strip()
                rank = _safe_int(rank_str)
                if not name or rank is None:
                    continue
                prob = rank_to_prob.get(rank, 0.0)
                self._last_name_list.append(name)
                self._last_name_weights.append(prob)

    def _read_gender_probability(self, filename: str) -> None:
        """
        File format:
            decade,gender,probability
            1950s,male,0.514
            ...
        """
        import os
        if not os.path.exists(filename):
            return
        with open(filename, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                decade = row["decade"].strip()
                gender = row["gender"].strip().lower()
                prob = _safe_float(row["probability"])
                if prob is None:
                    continue
                self._gender_prob.setdefault(decade, {})[gender] = prob

    # ------------------------------------------------------------------
    # Private: sampling
    # ------------------------------------------------------------------

    def _sample_gender(self, year_born: int) -> str:
        dec_str = _decade_str(year_born)
        probs = self._gender_prob.get(dec_str)
        if probs:
            male_prob = probs.get("male", 0.5)
            return "male" if random.random() < male_prob else "female"
        return "male" if random.random() < 0.5 else "female"

    def _sample_first_name(self, year_born: int, gender: str) -> str:
        dec_str = _decade_str(year_born)

        # CS 562: use gender_name_probability to decide if we assign a gendered name
        probs = self._gender_prob.get(dec_str, {})
        gender_prob = probs.get(gender, 1.0)
        effective_gender = gender if random.random() < gender_prob else "neutral"

        # Try gendered key first, then neutral/fallback
        for try_gender in [effective_gender, gender, "neutral"]:
            key = (dec_str, try_gender)
            if key in self._first_names:
                names, weights = self._first_names[key]
                return _weighted_choice(names, weights)

        # Fallback: any decade with same gender
        fallback_keys = [k for k in self._first_names if k[1] == gender]
        if fallback_keys:
            names, weights = self._first_names[fallback_keys[0]]
            return _weighted_choice(names, weights)

        return "Alex"

    def _sample_last_name(self) -> str:
        if not self._last_name_list:
            return "Smith"
        return _weighted_choice(self._last_name_list, self._last_name_weights)

    def _sample_year_died(self, year_born: int) -> int:
        dec = _decade_int(year_born)
        base_expectancy = self._life_expectancy.get(dec)

        if base_expectancy is None:
            # Use nearest available decade
            if self._life_expectancy:
                nearest = min(self._life_expectancy.keys(), key=lambda k: abs(k - dec))
                base_expectancy = self._life_expectancy[nearest]
            else:
                base_expectancy = 80.0

        lived = base_expectancy + random.uniform(-10.0, 10.0)
        lived = max(1.0, lived)
        return year_born + int(round(lived))


# ---------------------------------------------------------------------------
# FamilyTree: builds and queries the tree
# ---------------------------------------------------------------------------

class FamilyTree:
    """Drives tree generation and answers user queries."""

    MAX_YEAR = 2120

    def __init__(self, factory: PersonFactory) -> None:
        self.factory = factory
        self.all_people: List[Person] = []
        self._founder_last_names: List[str] = []

    def build(self, founder1_name: str, founder2_name: str) -> None:
        """Generate the full family tree starting from two founders born in 1950."""
        first1, last1 = _parse_full_name(founder1_name, default_first="Desmond", default_last="Jones")
        first2, last2 = _parse_full_name(founder2_name, default_first="Molly", default_last="Jones")

        founder1 = self.factory.get_person(year_born=1950, last_name=last1)
        founder1.first_name = first1

        founder2 = self.factory.get_person(year_born=1950, last_name=last2)
        founder2.first_name = first2

        founder1.set_partner(founder2)
        founder2.set_partner(founder1)

        self._founder_last_names = [last1, last2]
        self.all_people = [founder1, founder2]

        queue: List[Person] = [founder1]  # only process one of the pair to avoid duplicate children

        while queue:
            person = queue.pop(0)

            if person.year_born >= self.MAX_YEAR:
                continue

            # Try to give a partner to in-laws / non-founders if they don't have one
            if not person.has_partner():
                partner = self.factory.maybe_create_partner(person)
                if partner is not None:
                    person.set_partner(partner)
                    partner.set_partner(person)
                    self.all_people.append(partner)
                    # Don't add partner to queue — their children come from person

            # Generate children
            children = self.factory.create_children(person)
            for child in children:
                if child.year_born > self.MAX_YEAR:
                    continue
                # Direct descendants keep a founder last name
                child.last_name = random.choice(self._founder_last_names)
                person.add_child(child)
                self.all_people.append(child)
                queue.append(child)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def total_people(self) -> int:
        return len(self.all_people)

    def total_by_decade(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for person in self.all_people:
            dec = _decade_int(person.year_born)
            counts[dec] = counts.get(dec, 0) + 1
        return dict(sorted(counts.items()))

    def duplicate_names(self) -> List[str]:
        seen: Dict[str, int] = {}
        for person in self.all_people:
            name = person.full_name()
            seen[name] = seen.get(name, 0) + 1
        return sorted(name for name, count in seen.items() if count > 1)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _decade_int(year: int) -> int:
    """Return the decade as an int, e.g. 1957 -> 1950."""
    return (year // 10) * 10


def _decade_str(year: int) -> str:
    """Return decade as a string like '1950s'."""
    return f"{_decade_int(year)}s"


def _parse_decade_str(s: str) -> Optional[int]:
    """Parse '1950s' or '1950' into 1950."""
    cleaned = s.strip().rstrip("s")
    return _safe_int(cleaned)


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(float(s.strip()))
    except (ValueError, AttributeError):
        return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None


def _weighted_choice(items: List[str], weights: List[float]) -> str:
    total = sum(weights)
    if total <= 0:
        return random.choice(items)
    r = random.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= r:
            return item
    return items[-1]


def _spread_years(start: int, end: int, n: int) -> List[int]:
    """Return n years spread evenly between start and end (inclusive)."""
    if n == 1:
        return [round((start + end) / 2)]
    span = end - start
    return [round(start + i * span / (n - 1)) for i in range(n)]


def _parse_full_name(
    full: str,
    default_first: str = "Desmond",
    default_last: str = "Jones",
) -> Tuple[str, str]:
    parts = full.strip().split()
    if len(parts) == 0:
        return (default_first, default_last)
    if len(parts) == 1:
        return (parts[0], default_last)
    return (parts[0], parts[-1])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_menu() -> None:
    print()
    print("Are you interested in:")
    print("(T)otal number of people in the tree")
    print("Total number of people in the tree by (D)ecade")
    print("(N)ames duplicated")
    print("(Q)uit")


def main() -> None:
    print("Reading files...")
    factory = PersonFactory()

    try:
        factory.read_files()
    except FileNotFoundError as exc:
        print(f"ERROR: Missing required file: {exc.filename}")
        print("Make sure all CSV files are in the same directory as this script.")
        return
    except Exception as exc:
        print(f"ERROR reading files: {exc}")
        return

    print("Generating family tree...")

    founder1 = input("Enter founder 1 full name (default: Desmond Jones): ").strip()
    founder2 = input("Enter founder 2 full name (default: Molly Jones): ").strip()
    if not founder1:
        founder1 = "Desmond Jones"
    if not founder2:
        founder2 = "Molly Jones"

    tree = FamilyTree(factory=factory)

    try:
        tree.build(founder1, founder2)
    except Exception as exc:
        print(f"ERROR generating family tree: {exc}")
        return

    while True:
        _print_menu()
        choice = input("> ").strip().upper()

        if choice == "Q":
            print("Goodbye.")
            break
        elif choice == "T":
            print(f"The tree contains {tree.total_people()} people total")
        elif choice == "D":
            for decade, count in tree.total_by_decade().items():
                print(f"{decade}: {count}")
        elif choice == "N":
            duplicates = tree.duplicate_names()
            print(f"There are {len(duplicates)} duplicate names in the tree:")
            for name in duplicates:
                print(f"* {name}")
        else:
            print("Invalid option. Please enter T, D, N, or Q.")


if __name__ == "__main__":
    main()