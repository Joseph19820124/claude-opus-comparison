#!/usr/bin/env python3
"""
Benchmark: Claude Opus 4.5 vs 4.6 Comparison
Runs 35 tasks across 13 categories and captures metrics.
"""

import json
import subprocess
import time
import re
import sys
from datetime import datetime

# ─── Task Definitions ────────────────────────────────────────────────────────

TASKS = [
    # ── 1. Math & Arithmetic ──────────────────────────────────────────────
    {
        "id": "math_1",
        "category": "Math & Arithmetic",
        "name": "Multi-step Word Problem",
        "prompt": (
            "A store sells notebooks for $3 each and pens for $1.50 each. "
            "Maria buys 4 notebooks and 6 pens. She pays with a $50 bill. "
            "She then uses her change to buy as many erasers as possible at $2.75 each. "
            "How many erasers can she buy and how much money is left over? "
            "Show your work step by step, then give the final answer as: "
            "ANSWER: X erasers, $Y.YY remaining"
        ),
        "expected_answer": "ANSWER: 10 erasers, $1.50 remaining",
        "scoring": "exact_match",
        "match_pattern": r"10 erasers.*\$1\.50"
    },
    {
        "id": "math_2",
        "category": "Math & Arithmetic",
        "name": "Probability Question",
        "prompt": (
            "A bag contains 5 red balls, 3 blue balls, and 2 green balls. "
            "You draw 2 balls without replacement. "
            "What is the probability that both balls are the same color? "
            "Express the answer as a simplified fraction. "
            "ANSWER: (fraction)"
        ),
        "expected_answer": "14/45",
        "scoring": "exact_match",
        "match_pattern": r"14/45"
    },
    {
        "id": "math_3",
        "category": "Math & Arithmetic",
        "name": "Algebra / Equation Solving",
        "prompt": (
            "Solve the system of equations:\n"
            "  2x + 3y = 12\n"
            "  4x - y = 5\n"
            "Give the answer as ANSWER: x = ?, y = ?"
        ),
        "expected_answer": "x = 27/14, y = 25/7",
        "scoring": "exact_match",
        "match_pattern": r"x\s*=\s*(27/14|1\.928|1\.93)"
    },

    # ── 2. Logic & Reasoning ─────────────────────────────────────────────
    {
        "id": "logic_1",
        "category": "Logic & Reasoning",
        "name": "Knights & Knaves",
        "prompt": (
            "On an island, knights always tell the truth and knaves always lie. "
            "You meet three people: A, B, and C.\n"
            "A says: 'B is a knave.'\n"
            "B says: 'A and C are the same type (both knights or both knaves).'\n"
            "C says: 'I am a knight.'\n"
            "Determine the type of each person. "
            "ANSWER: A is a ___, B is a ___, C is a ___"
        ),
        "expected_answer": "A=knight,B=knave,C=knave OR A=knave,B=knight,C=knave",
        "scoring": "logic_check",
        "match_pattern": r"(A.*knight.*B.*knave.*C.*knave|A.*knave.*B.*knight.*C.*knave)"
    },
    {
        "id": "logic_2",
        "category": "Logic & Reasoning",
        "name": "Constraint Satisfaction",
        "prompt": (
            "Five houses in a row are painted different colors: red, green, blue, yellow, white. "
            "Each house has a person of different nationality: American, British, Canadian, Danish, Egyptian.\n"
            "Clues:\n"
            "1. The British person lives in the red house.\n"
            "2. The green house is immediately to the right of the white house.\n"
            "3. The Danish person lives in the yellow house.\n"
            "4. The American lives in the first house.\n"
            "5. The Canadian lives next to the blue house.\n"
            "6. The Egyptian lives in the fifth house.\n"
            "7. The green house is not the first or fifth house.\n"
            "What color is each house from left to right? "
            "ANSWER: (list 5 colors left to right)"
        ),
        "expected_answer": "yellow, white, green, red, blue",
        "scoring": "constraint_check",
        "match_pattern": r"yellow.*white.*green.*red.*blue"
    },
    {
        "id": "logic_3",
        "category": "Logic & Reasoning",
        "name": "Sequence Pattern Recognition",
        "prompt": (
            "What is the next number in this sequence?\n"
            "2, 6, 14, 30, 62, ?\n"
            "Explain the pattern, then give ANSWER: (number)"
        ),
        "expected_answer": "126",
        "scoring": "exact_match",
        "match_pattern": r"126"
    },

    # ── 3. Coding - Generation ───────────────────────────────────────────
    {
        "id": "code_gen_1",
        "category": "Coding - Generation",
        "name": "LRU Cache Implementation",
        "prompt": (
            "Implement an LRU Cache class in Python with the following interface:\n"
            "- `LRUCache(capacity)`: Initialize with positive size capacity.\n"
            "- `get(key)`: Return the value if key exists, otherwise return -1.\n"
            "- `put(key, value)`: Update or insert. Evict least recently used if at capacity.\n"
            "Both operations must be O(1) time complexity.\n"
            "Output ONLY the Python code, no explanation."
        ),
        "expected_answer": "working LRU cache class",
        "scoring": "code_execution",
        "test_code": """
import sys, io
# Exec the generated code
exec(GENERATED_CODE)
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1, f"Expected 1, got {cache.get(1)}"
cache.put(3, 3)  # evicts key 2
assert cache.get(2) == -1, f"Expected -1, got {cache.get(2)}"
cache.put(4, 4)  # evicts key 1
assert cache.get(1) == -1, f"Expected -1, got {cache.get(1)}"
assert cache.get(3) == 3, f"Expected 3, got {cache.get(3)}"
assert cache.get(4) == 4, f"Expected 4, got {cache.get(4)}"
print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "code_gen_2",
        "category": "Coding - Generation",
        "name": "N-Queens Solver",
        "prompt": (
            "Write a Python function `solve_n_queens(n)` that returns the number of distinct "
            "solutions to the N-Queens puzzle. For example, solve_n_queens(4) should return 2, "
            "solve_n_queens(8) should return 92.\n"
            "Output ONLY the Python code, no explanation."
        ),
        "expected_answer": "working n-queens solver",
        "scoring": "code_execution",
        "test_code": """
exec(GENERATED_CODE)
assert solve_n_queens(1) == 1, f"n=1: expected 1, got {solve_n_queens(1)}"
assert solve_n_queens(4) == 2, f"n=4: expected 2, got {solve_n_queens(4)}"
assert solve_n_queens(8) == 92, f"n=8: expected 92, got {solve_n_queens(8)}"
print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "code_gen_3",
        "category": "Coding - Generation",
        "name": "Email Validation Regex",
        "prompt": (
            "Write a Python function `is_valid_email(email)` that uses a regex to validate "
            "email addresses. It should accept standard emails like user@example.com, "
            "user.name+tag@domain.co.uk, but reject invalid ones like @domain.com, "
            "user@, user@.com, user@domain..com.\n"
            "Output ONLY the Python code, no explanation."
        ),
        "expected_answer": "working email validator",
        "scoring": "code_execution",
        "test_code": """
exec(GENERATED_CODE)
valid = ["user@example.com", "user.name+tag@domain.co.uk", "test123@sub.domain.com"]
invalid = ["@domain.com", "user@", "user@.com", "user@domain..com", "", "no-at-sign", "user @domain.com"]
for e in valid:
    assert is_valid_email(e), f"Should be valid: {e}"
for e in invalid:
    assert not is_valid_email(e), f"Should be invalid: {e}"
print("ALL_TESTS_PASSED")
"""
    },

    # ── 4. Coding - Debugging ────────────────────────────────────────────
    {
        "id": "code_debug_1",
        "category": "Coding - Debugging",
        "name": "Find Off-by-One Bug",
        "prompt": (
            "The following Python function is supposed to find the k-th smallest element "
            "in a sorted matrix (each row and column is sorted). It has a bug. "
            "Find the bug and provide the corrected code.\n\n"
            "```python\n"
            "def kth_smallest(matrix, k):\n"
            "    n = len(matrix)\n"
            "    lo, hi = matrix[0][0], matrix[n-1][n-1]\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        count = 0\n"
            "        j = n - 1\n"
            "        for i in range(n):\n"
            "            while j >= 0 and matrix[i][j] > mid:\n"
            "                j -= 1\n"
            "            count += j + 1\n"
            "        if count < k:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid\n"
            "    return lo\n"
            "```\n\n"
            "Output ONLY the corrected Python code, no explanation."
        ),
        "expected_answer": "j should be reset inside the for loop",
        "scoring": "code_execution",
        "test_code": """
exec(GENERATED_CODE)
m1 = [[1,5,9],[10,11,13],[12,13,15]]
assert kth_smallest(m1, 1) == 1, f"Expected 1, got {kth_smallest(m1, 1)}"
assert kth_smallest(m1, 5) == 11, f"Expected 11, got {kth_smallest(m1, 5)}"
assert kth_smallest(m1, 8) == 13, f"Expected 13, got {kth_smallest(m1, 8)}"
m2 = [[1,2],[1,3]]
assert kth_smallest(m2, 3) == 2, f"Expected 2, got {kth_smallest(m2, 3)}"
print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "code_debug_2",
        "category": "Coding - Debugging",
        "name": "Fix Race Condition",
        "prompt": (
            "The following Python code has a race condition in the counter. "
            "Fix it so the final count is always exactly 10000.\n\n"
            "```python\n"
            "import threading\n\n"
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n\n"
            "    def increment(self):\n"
            "        self.count += 1\n\n"
            "def worker(counter, n):\n"
            "    for _ in range(n):\n"
            "        counter.increment()\n\n"
            "def main():\n"
            "    counter = Counter()\n"
            "    threads = []\n"
            "    for _ in range(10):\n"
            "        t = threading.Thread(target=worker, args=(counter, 1000))\n"
            "        threads.append(t)\n"
            "        t.start()\n"
            "    for t in threads:\n"
            "        t.join()\n"
            "    return counter.count\n"
            "```\n\n"
            "Output ONLY the corrected Python code, no explanation."
        ),
        "expected_answer": "add threading.Lock",
        "scoring": "code_execution",
        "test_code": """
exec(GENERATED_CODE)
result = main()
assert result == 10000, f"Expected 10000, got {result}"
# Run multiple times to be sure
for i in range(5):
    r = main()
    assert r == 10000, f"Run {i}: Expected 10000, got {r}"
print("ALL_TESTS_PASSED")
"""
    },

    # ── 5. Instruction Following ─────────────────────────────────────────
    {
        "id": "instruct_1",
        "category": "Instruction Following",
        "name": "Strict JSON Schema",
        "prompt": (
            "Generate a JSON object describing a fictional book with EXACTLY these fields:\n"
            '- "title": string (5-10 words)\n'
            '- "author": string (first and last name)\n'
            '- "year": integer (between 2020 and 2025)\n'
            '- "genres": array of exactly 3 strings\n'
            '- "rating": number (between 1.0 and 5.0, one decimal place)\n'
            '- "available": boolean\n'
            "Output ONLY the JSON, no other text."
        ),
        "expected_answer": "valid JSON with exact schema",
        "scoring": "json_schema",
        "schema_check": {
            "required_fields": ["title", "author", "year", "genres", "rating", "available"],
            "year_range": [2020, 2025],
            "genres_count": 3,
            "rating_range": [1.0, 5.0]
        }
    },
    {
        "id": "instruct_2",
        "category": "Instruction Following",
        "name": "Multi-Constraint Format",
        "prompt": (
            "Write a response that follows ALL of these constraints:\n"
            "1. Exactly 3 paragraphs\n"
            "2. Each paragraph must be exactly 2 sentences\n"
            "3. The first word of each paragraph must start with the letters A, B, C respectively\n"
            "4. The topic must be about space exploration\n"
            "5. No paragraph may exceed 50 words\n"
            "Output ONLY the paragraphs, no other text."
        ),
        "expected_answer": "response following all constraints",
        "scoring": "instruction_check",
        "constraints": ["3_paragraphs", "2_sentences_each", "ABC_starts", "space_topic", "50_word_limit"]
    },
    {
        "id": "instruct_3",
        "category": "Instruction Following",
        "name": "Exact Word Count",
        "prompt": (
            "Write a description of machine learning in EXACTLY 25 words. "
            "Not 24, not 26 - exactly 25 words. "
            "Output ONLY the description."
        ),
        "expected_answer": "exactly 25 words",
        "scoring": "word_count",
        "target_words": 25
    },

    # ── 6. Knowledge & Analysis ──────────────────────────────────────────
    {
        "id": "knowledge_1",
        "category": "Knowledge & Analysis",
        "name": "Architecture Comparison",
        "prompt": (
            "Compare microservices vs monolithic architecture across exactly 5 dimensions: "
            "scalability, development speed, deployment complexity, debugging ease, and cost. "
            "For each dimension, give a 1-sentence verdict stating which approach wins and why. "
            "Format as a numbered list. No other text."
        ),
        "expected_answer": "accurate technical comparison",
        "scoring": "structural_check",
        "structural_checks": {
            "numbered_items": 5,
            "required_keywords": ["scalability", "development speed", "deployment", "debugging", "cost"],
            "min_length": 100,
        }
    },
    {
        "id": "knowledge_2",
        "category": "Knowledge & Analysis",
        "name": "Scientific Explanation",
        "prompt": (
            "Explain how mRNA vaccines work in exactly 4 steps. "
            "Each step should be one sentence. Number the steps. "
            "Use simple language a high school student would understand. "
            "Output ONLY the 4 numbered steps."
        ),
        "expected_answer": "accurate 4-step explanation",
        "scoring": "structural_check",
        "structural_checks": {
            "numbered_items": 4,
            "required_keywords": ["mRNA", "protein", "immune"],
            "min_length": 80,
        }
    },

    # ── 7. Creative & Language ───────────────────────────────────────────
    {
        "id": "creative_1",
        "category": "Creative & Language",
        "name": "Concise Analogy",
        "prompt": (
            "Create a single-sentence analogy that explains recursion to a 10-year-old. "
            "The analogy must reference a real-world object or activity. "
            "Maximum 30 words. Output ONLY the analogy."
        ),
        "expected_answer": "clear, creative analogy",
        "scoring": "structural_check",
        "structural_checks": {
            "max_sentences": 1,
            "max_words": 30,
            "min_length": 20,
        }
    },
    {
        "id": "creative_2",
        "category": "Creative & Language",
        "name": "Precise Summarization",
        "prompt": (
            "Summarize the following passage in EXACTLY 2 sentences:\n\n"
            "The Internet of Things (IoT) refers to the network of physical devices, "
            "vehicles, home appliances, and other items embedded with electronics, software, "
            "sensors, and connectivity which enables these things to connect, collect and "
            "exchange data. The IoT allows objects to be sensed or controlled remotely across "
            "existing network infrastructure, creating opportunities for more direct integration "
            "of the physical world into computer-based systems. This results in improved efficiency, "
            "accuracy, and economic benefit in addition to reduced human intervention. IoT devices "
            "are a part of the larger concept of home automation. Large segments of IoT devices "
            "are created for consumer use, including connected vehicles, home automation, wearable "
            "technology, connected health, and appliances.\n\n"
            "Output ONLY the 2-sentence summary."
        ),
        "expected_answer": "accurate 2-sentence summary",
        "scoring": "structural_check",
        "structural_checks": {
            "exact_sentences": 2,
            "required_keywords": ["IoT"],
            "min_length": 40,
            "max_words": 80,
        }
    },

    # ── 8. Multi-step Planning ───────────────────────────────────────────
    {
        "id": "planning_1",
        "category": "Multi-step Planning",
        "name": "System Design",
        "prompt": (
            "Design a URL shortener service. Cover these aspects in order:\n"
            "1. API endpoints (2-3 endpoints)\n"
            "2. Database schema (key fields only)\n"
            "3. Key algorithm for generating short codes\n"
            "4. How to handle high traffic (1 strategy)\n"
            "5. One potential issue and its mitigation\n"
            "Keep each section to 2-3 sentences max. Use numbered sections."
        ),
        "expected_answer": "complete system design covering all 5 aspects",
        "scoring": "structural_check",
        "structural_checks": {
            "numbered_items": 5,
            "required_keywords": ["endpoint", "database", "algorithm", "traffic", "issue"],
            "min_length": 200,
        }
    },
    {
        "id": "planning_2",
        "category": "Multi-step Planning",
        "name": "Algorithm Tradeoff Analysis",
        "prompt": (
            "You need to implement a search feature for a product catalog with 10 million items. "
            "Compare exactly 3 approaches:\n"
            "1. SQL LIKE queries\n"
            "2. Elasticsearch\n"
            "3. In-memory trie\n"
            "For each, state: time complexity for search, memory usage (high/medium/low), "
            "and the main tradeoff in one sentence. "
            "Then state which you'd choose and why in one sentence. "
            "Format as a numbered list with a final recommendation."
        ),
        "expected_answer": "accurate comparison with clear recommendation",
        "scoring": "structural_check",
        "structural_checks": {
            "numbered_items": 3,
            "required_keywords": ["SQL", "Elasticsearch", "trie", "recommend"],
            "min_length": 150,
        }
    },

    # ── 9. Harder Math ────────────────────────────────────────────────
    {
        "id": "math_hard_1",
        "category": "Harder Math",
        "name": "Combinatorics — Restricted Permutations",
        "prompt": (
            "How many 4-digit numbers can be formed using the digits 1, 2, 3, 4, 5 "
            "(repetition allowed) such that the number is divisible by 4? "
            "Show your work, then give ANSWER: (number)"
        ),
        "expected_answer": "125",
        "scoring": "exact_match",
        "match_pattern": r"(?:ANSWER[:\s]*)?125\b"
    },
    {
        "id": "math_hard_2",
        "category": "Harder Math",
        "name": "Modular Arithmetic Chain",
        "prompt": (
            "Compute the following step by step:\n"
            "Let a = 7^3 mod 13\n"
            "Let b = (a * 11) mod 13\n"
            "Let c = b^2 mod 13\n"
            "Let d = (c + 5) mod 13\n"
            "What is d? Show all intermediate values, then give ANSWER: (number)"
        ),
        "expected_answer": "1",
        "scoring": "exact_match",
        "match_pattern": r"ANSWER[:\s]*1\b"
    },
    {
        "id": "math_hard_3",
        "category": "Harder Math",
        "name": "Optimization — Knapsack",
        "prompt": (
            "You have a knapsack that can hold at most 15 kg. "
            "You have these items (each can only be taken once):\n"
            "  A: weight=3kg, value=$4\n"
            "  B: weight=4kg, value=$5\n"
            "  C: weight=5kg, value=$7\n"
            "  D: weight=7kg, value=$9\n"
            "  E: weight=2kg, value=$3\n"
            "  F: weight=6kg, value=$8\n"
            "What is the maximum total value you can carry? "
            "List the items you'd pick and the total. "
            "ANSWER: $XX"
        ),
        "expected_answer": "$20",
        "scoring": "exact_match",
        "match_pattern": r"\$20\b"
    },

    # ── 10. Harder Logic ──────────────────────────────────────────────
    {
        "id": "logic_hard_1",
        "category": "Harder Logic",
        "name": "River Crossing Puzzle",
        "prompt": (
            "Solve the classic river crossing puzzle:\n"
            "A farmer must transport a wolf, a goat, and a cabbage across a river. "
            "The boat can carry the farmer and ONE item. "
            "If left alone: the wolf eats the goat, the goat eats the cabbage. "
            "The farmer must be present to prevent eating.\n"
            "Give a valid sequence of crossings. For each step, state what the farmer "
            "takes across and the direction (over or back). "
            "Format each step as: 'Step N: Take X over/back' "
            "Number all steps."
        ),
        "expected_answer": "valid 7-step crossing sequence",
        "scoring": "code_execution",
        "test_code": """
import re

response = GENERATED_CODE

# Parse steps from the response
steps = re.findall(r'[Ss]tep\\s*\\d+[:\\s]+(.*?)(?=\\n|$)', response)
if not steps:
    steps = re.findall(r'\\d+[.\\):]\\s*(.*?)(?=\\n|$)', response)

# Simulate the crossing
left = {'wolf', 'goat', 'cabbage'}
right = set()
farmer_on_left = True
valid = True
error_msg = ""

for i, step in enumerate(steps):
    step_lower = step.lower()
    # Determine what is being moved
    item = None
    for candidate in ['wolf', 'goat', 'cabbage']:
        if candidate in step_lower:
            item = candidate
            break

    going_over = 'over' in step_lower or 'right' in step_lower or 'across' in step_lower
    going_back = 'back' in step_lower or 'return' in step_lower or 'left' in step_lower

    if item is None and ('nothing' in step_lower or 'alone' in step_lower or 'empty' in step_lower):
        # Farmer crosses alone
        farmer_on_left = not farmer_on_left
    elif item:
        if farmer_on_left and item in left:
            left.remove(item)
            right.add(item)
            farmer_on_left = False
        elif not farmer_on_left and item in right:
            right.remove(item)
            left.add(item)
            farmer_on_left = True
        else:
            error_msg = f"Step {i+1}: item '{item}' not on farmer's side"
            valid = False
            break
    else:
        continue

    # Check constraints (farmer not present side)
    danger_side = left if not farmer_on_left else right
    if 'wolf' in danger_side and 'goat' in danger_side:
        error_msg = f"Step {i+1}: wolf eats goat on {'left' if not farmer_on_left else 'right'}"
        valid = False
        break
    if 'goat' in danger_side and 'cabbage' in danger_side:
        error_msg = f"Step {i+1}: goat eats cabbage on {'left' if not farmer_on_left else 'right'}"
        valid = False
        break

if valid and right == {'wolf', 'goat', 'cabbage'}:
    print("ALL_TESTS_PASSED")
elif not valid:
    assert False, f"Invalid: {error_msg}"
else:
    assert False, f"Not all items crossed. Left: {left}, Right: {right}"
"""
    },
    {
        "id": "logic_hard_2",
        "category": "Harder Logic",
        "name": "Boolean Truth Table",
        "prompt": (
            "Generate the complete truth table for the boolean expression:\n"
            "  (A AND B) OR (NOT A AND C)\n"
            "Use columns: A, B, C, Result\n"
            "Use 0 and 1 for values. List all 8 rows.\n"
            "Output ONLY the truth table with a header row."
        ),
        "expected_answer": "correct 8-row truth table",
        "scoring": "code_execution",
        "test_code": """
import re

response = GENERATED_CODE

# Extract rows with 4 binary values
rows = re.findall(r'([01])\\s+([01])\\s+([01])\\s+([01])', response)
# Also try pipe-delimited format
if len(rows) < 8:
    rows = re.findall(r'([01])\\s*\\|\\s*([01])\\s*\\|\\s*([01])\\s*\\|\\s*([01])', response)

assert len(rows) >= 8, f"Expected 8 rows, found {len(rows)}"

# Verify each row
expected = {}
for a in range(2):
    for b in range(2):
        for c in range(2):
            result = (a and b) or (not a and c)
            expected[(a, b, c)] = int(result)

found_combos = set()
for row in rows[:8]:
    a, b, c, r = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    combo = (a, b, c)
    if combo in found_combos:
        continue
    found_combos.add(combo)
    exp = expected[combo]
    assert r == exp, f"Row A={a},B={b},C={c}: expected {exp}, got {r}"

assert len(found_combos) == 8, f"Expected 8 unique combos, found {len(found_combos)}"
print("ALL_TESTS_PASSED")
"""
    },

    # ── 11. Harder Coding ─────────────────────────────────────────────
    {
        "id": "code_hard_1",
        "category": "Harder Coding",
        "name": "Trie with Autocomplete",
        "prompt": (
            "Implement a Trie class in Python with these methods:\n"
            "- `insert(word)`: Insert a word into the trie.\n"
            "- `search(word)`: Return True if word is in the trie, False otherwise.\n"
            "- `starts_with(prefix)`: Return True if any word starts with prefix.\n"
            "- `autocomplete(prefix)`: Return a sorted list of all words that start with prefix.\n"
            "Output ONLY the Python code, no explanation."
        ),
        "expected_answer": "working Trie with autocomplete",
        "scoring": "code_execution",
        "test_code": """
exec(GENERATED_CODE)
t = Trie()
words = ["apple", "app", "application", "apply", "banana", "band", "ban"]
for w in words:
    t.insert(w)

# search
assert t.search("apple") == True, "apple should be found"
assert t.search("app") == True, "app should be found"
assert t.search("ap") == False, "ap should not be found"
assert t.search("banana") == True, "banana should be found"
assert t.search("bananas") == False, "bananas should not be found"

# starts_with
assert t.starts_with("app") == True, "prefix app should exist"
assert t.starts_with("bz") == False, "prefix bz should not exist"
assert t.starts_with("ban") == True, "prefix ban should exist"

# autocomplete
result = t.autocomplete("app")
assert result == ["app", "apple", "application", "apply"], f"autocomplete('app') = {result}"
result = t.autocomplete("ban")
assert result == ["ban", "banana", "band"], f"autocomplete('ban') = {result}"
result = t.autocomplete("xyz")
assert result == [], f"autocomplete('xyz') = {result}"

# Edge cases
t2 = Trie()
t2.insert("")
assert t2.autocomplete("") == [""] or t2.autocomplete("") == [], "empty string edge case"
print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "code_hard_2",
        "category": "Harder Coding",
        "name": "Topological Sort with Cycle Detection",
        "prompt": (
            "Write a Python function `topological_sort(graph)` where graph is a dict mapping "
            "each node to a list of its dependencies (nodes it depends on).\n"
            "Example: {'a': ['b', 'c'], 'b': ['c'], 'c': []} means 'a' depends on 'b' and 'c'.\n"
            "Return a list of nodes in valid topological order (dependencies before dependents).\n"
            "If a cycle is detected, raise a ValueError with the message 'cycle detected'.\n"
            "Output ONLY the Python code, no explanation."
        ),
        "expected_answer": "working topological sort with cycle detection",
        "scoring": "code_execution",
        "test_code": """
exec(GENERATED_CODE)

# Test 1: simple DAG
g1 = {'a': ['b', 'c'], 'b': ['c'], 'c': []}
r1 = topological_sort(g1)
assert 'c' in r1 and 'b' in r1 and 'a' in r1, f"Missing nodes: {r1}"
assert r1.index('c') < r1.index('b'), f"c should come before b: {r1}"
assert r1.index('b') < r1.index('a'), f"b should come before a: {r1}"

# Test 2: larger DAG
g2 = {'d': ['b', 'c'], 'c': ['a'], 'b': ['a'], 'a': []}
r2 = topological_sort(g2)
assert r2.index('a') < r2.index('b'), f"a should come before b: {r2}"
assert r2.index('a') < r2.index('c'), f"a should come before c: {r2}"
assert r2.index('b') < r2.index('d'), f"b should come before d: {r2}"
assert r2.index('c') < r2.index('d'), f"c should come before d: {r2}"

# Test 3: cycle detection
g3 = {'a': ['b'], 'b': ['c'], 'c': ['a']}
try:
    topological_sort(g3)
    assert False, "Should have raised ValueError for cycle"
except ValueError as e:
    assert 'cycle' in str(e).lower(), f"Error message should mention cycle: {e}"

# Test 4: single node
g4 = {'x': []}
r4 = topological_sort(g4)
assert r4 == ['x'], f"Single node: {r4}"

# Test 5: self-loop
g5 = {'a': ['a']}
try:
    topological_sort(g5)
    assert False, "Should have raised ValueError for self-loop"
except ValueError as e:
    assert 'cycle' in str(e).lower(), f"Error message should mention cycle: {e}"

print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "code_hard_3",
        "category": "Harder Coding",
        "name": "Optimize Brute Force — Two Sum to O(n)",
        "prompt": (
            "The following Python function finds two numbers in a list that add up to a target. "
            "It works but is O(n^2). Rewrite it to be O(n) using a hash map approach.\n\n"
            "```python\n"
            "def two_sum(nums, target):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i + 1, len(nums)):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return (i, j)\n"
            "    return None\n"
            "```\n\n"
            "The optimized version must:\n"
            "1. Return a tuple of two indices (i, j) where i < j\n"
            "2. Handle the case where no solution exists (return None)\n"
            "3. Run in O(n) time\n"
            "Output ONLY the Python code, no explanation."
        ),
        "expected_answer": "O(n) two_sum using hash map",
        "scoring": "code_execution",
        "test_code": """
import time
exec(GENERATED_CODE)

# Correctness tests
assert two_sum([2, 7, 11, 15], 9) == (0, 1), f"Test 1 failed: {two_sum([2, 7, 11, 15], 9)}"
assert two_sum([3, 2, 4], 6) == (1, 2), f"Test 2 failed: {two_sum([3, 2, 4], 6)}"
assert two_sum([3, 3], 6) == (0, 1), f"Test 3 failed: {two_sum([3, 3], 6)}"
assert two_sum([1, 2, 3], 10) is None, f"Test 4 failed: {two_sum([1, 2, 3], 10)}"

# Performance test: O(n) should handle 1M elements quickly
large = list(range(500000))
large.append(999999)  # target pair
start = time.time()
result = two_sum(large, 500000 + 999999 - 1)
elapsed = time.time() - start
# O(n) should be < 2s, O(n^2) would take minutes
assert elapsed < 3.0, f"Too slow ({elapsed:.1f}s) - likely not O(n)"
assert result is not None, "Should find pair in large list"

print("ALL_TESTS_PASSED")
"""
    },

    # ── 12. Precision Instruction Following ───────────────────────────
    {
        "id": "instruct_acro_1",
        "category": "Precision Instruction Following",
        "name": "Acronym Constraint — CLAUDE",
        "prompt": (
            "Write exactly 6 sentences about artificial intelligence. "
            "The FIRST LETTER of each sentence must spell out C-L-A-U-D-E in order. "
            "Each sentence must be about AI. "
            "Output ONLY the 6 sentences, one per line."
        ),
        "expected_answer": "6 sentences with first letters spelling CLAUDE",
        "scoring": "code_execution",
        "test_code": """
response = GENERATED_CODE

lines = [l.strip() for l in response.strip().split('\\n') if l.strip()]
assert len(lines) >= 6, f"Expected 6 sentences/lines, got {len(lines)}"

target = "CLAUDE"
for i, letter in enumerate(target):
    first_char = lines[i][0].upper() if lines[i] else ''
    assert first_char == letter, f"Line {i+1} starts with '{first_char}', expected '{letter}': {lines[i][:40]}"

print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "instruct_reverse_1",
        "category": "Precision Instruction Following",
        "name": "Reverse Alphabetical Paragraphs",
        "prompt": (
            "Write exactly 4 paragraphs about climate change. "
            "The first word of each paragraph must start with these letters IN ORDER: D, C, B, A. "
            "Each paragraph should be 2-3 sentences. "
            "Output ONLY the 4 paragraphs separated by blank lines."
        ),
        "expected_answer": "4 paragraphs starting with D, C, B, A",
        "scoring": "code_execution",
        "test_code": """
response = GENERATED_CODE

paragraphs = [p.strip() for p in response.strip().split('\\n\\n') if p.strip()]
assert len(paragraphs) >= 4, f"Expected 4 paragraphs, got {len(paragraphs)}"

target = "DCBA"
for i, letter in enumerate(target):
    first_char = paragraphs[i].split()[0][0].upper() if paragraphs[i].split() else ''
    assert first_char == letter, f"Paragraph {i+1} starts with '{first_char}', expected '{letter}'"

print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "instruct_charcount_1",
        "category": "Precision Instruction Following",
        "name": "Exact Character Count",
        "prompt": (
            "Write a description of the ocean in EXACTLY 200 characters (including spaces and punctuation). "
            "Not 199, not 201 - exactly 200 characters. "
            "Output ONLY the description, nothing else."
        ),
        "expected_answer": "exactly 200 characters",
        "scoring": "code_execution",
        "test_code": """
response = GENERATED_CODE
text = response.strip()
length = len(text)
if length == 200:
    print("ALL_TESTS_PASSED")
else:
    diff = length - 200
    assert False, f"Expected 200 characters, got {length} (off by {diff:+d})"
"""
    },
    {
        "id": "instruct_noletter_1",
        "category": "Precision Instruction Following",
        "name": "No-Letter-E Constraint",
        "prompt": (
            "Write a 3-sentence explanation of how a car engine works. "
            "CONSTRAINT: Do NOT use the letter 'e' (uppercase or lowercase) anywhere in your response. "
            "This is a lipogram challenge. "
            "Output ONLY the 3 sentences."
        ),
        "expected_answer": "3 sentences with no letter e",
        "scoring": "code_execution",
        "test_code": """
response = GENERATED_CODE
text = response.strip()

# Check for letter 'e'
e_positions = [i for i, c in enumerate(text) if c.lower() == 'e']
assert len(e_positions) == 0, f"Found {len(e_positions)} letter 'e' at positions: {e_positions[:10]}"

# Check it's not empty and has content
assert len(text) > 30, f"Response too short ({len(text)} chars)"

print("ALL_TESTS_PASSED")
"""
    },

    # ── 13. Tricky Edge Cases ─────────────────────────────────────────
    {
        "id": "edge_math_trap",
        "category": "Tricky Edge Cases",
        "name": "Floating Point Trap — 0.1 + 0.2",
        "prompt": (
            "What is 0.1 + 0.2? "
            "Give the mathematically correct answer, not a floating point artifact. "
            "ANSWER: (number)"
        ),
        "expected_answer": "0.3",
        "scoring": "exact_match",
        "match_pattern": r"(?:ANSWER[:\s]*)?\b0\.3\b(?!0)"
    },
    {
        "id": "edge_self_ref",
        "category": "Tricky Edge Cases",
        "name": "Self-Referential Word Count",
        "prompt": (
            "Write a sentence that contains EXACTLY 10 words. "
            "The sentence should be about the weather. "
            "Output ONLY the sentence."
        ),
        "expected_answer": "a sentence with exactly 10 words",
        "scoring": "code_execution",
        "test_code": """
response = GENERATED_CODE
text = response.strip().rstrip('.')
# Count words
words = text.split()
word_count = len(words)
assert word_count == 10, f"Expected exactly 10 words, got {word_count}: {words}"
print("ALL_TESTS_PASSED")
"""
    },
    {
        "id": "edge_negation",
        "category": "Tricky Edge Cases",
        "name": "Negation Test — Non-Flying Mammals",
        "prompt": (
            "List exactly 5 mammals that do NOT fly. "
            "Just the names, one per line. "
            "Output ONLY the 5 names."
        ),
        "expected_answer": "5 non-flying mammals (no bats)",
        "scoring": "code_execution",
        "test_code": """
response = GENERATED_CODE
lines = [l.strip().lower().rstrip('.') for l in response.strip().split('\\n') if l.strip()]
# Remove numbering like "1. " or "- "
import re
cleaned = []
for line in lines:
    line = re.sub(r'^[\\d]+[.\\)\\s]+', '', line).strip()
    line = re.sub(r'^[-*]\\s+', '', line).strip()
    cleaned.append(line)

assert len(cleaned) >= 5, f"Expected 5 items, got {len(cleaned)}"

# Check no flying mammals (bats)
flying_keywords = ['bat', 'flying fox', 'fruit bat', 'vampire bat', 'colugo', 'flying squirrel']
for item in cleaned[:5]:
    for fk in flying_keywords:
        assert fk not in item, f"'{item}' can fly or glide! Should list non-flying mammals only."

print("ALL_TESTS_PASSED")
"""
    },
]


# ─── Scoring Functions ───────────────────────────────────────────────────────

def score_exact_match(response, task):
    """Score based on regex pattern match."""
    pattern = task.get("match_pattern", "")
    if re.search(pattern, response, re.IGNORECASE):
        return 1.0
    return 0.0


def score_code_execution(response, task):
    """Extract code from response and run test cases."""
    # Extract python code from markdown fences if present
    code = response
    code_match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1)

    test_code = task["test_code"].replace("GENERATED_CODE", repr(code))

    try:
        result = subprocess.run(
            ["python3", "-c", test_code],
            capture_output=True, text=True, timeout=30
        )
        if "ALL_TESTS_PASSED" in result.stdout:
            return 1.0
        elif result.returncode == 0:
            return 0.7
        else:
            # Check for partial passes
            error_msg = result.stderr + result.stdout
            if "AssertionError" in error_msg or "AssertionError" in error_msg:
                return 0.3
            return 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:
        return 0.0


def score_json_schema(response, task):
    """Validate JSON output against schema requirements."""
    score = 1.0
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if not json_match:
            # Try the whole response
            data = json.loads(response.strip())
        else:
            data = json.loads(json_match.group())

        schema = task["schema_check"]

        # Check required fields
        for field in schema["required_fields"]:
            if field not in data:
                score -= 0.15

        # Check year range
        if "year" in data:
            yr = data["year"]
            if not (schema["year_range"][0] <= yr <= schema["year_range"][1]):
                score -= 0.1

        # Check genres count
        if "genres" in data:
            if len(data["genres"]) != schema["genres_count"]:
                score -= 0.1

        # Check rating range
        if "rating" in data:
            r = data["rating"]
            if not (schema["rating_range"][0] <= r <= schema["rating_range"][1]):
                score -= 0.1

        # Check types
        if "available" in data and not isinstance(data["available"], bool):
            score -= 0.1

        return max(0, score)
    except (json.JSONDecodeError, Exception):
        return 0.0


def score_instruction_check(response, task):
    """Check multiple instruction constraints."""
    score = 1.0
    text = response.strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # 3 paragraphs
    if len(paragraphs) != 3:
        score -= 0.25

    # 2 sentences each
    for p in paragraphs[:3]:
        sentences = [s.strip() for s in re.split(r'[.!?]+', p) if s.strip()]
        if len(sentences) != 2:
            score -= 0.1

    # ABC starts
    if len(paragraphs) >= 1 and not paragraphs[0][0].upper() == 'A':
        score -= 0.15
    if len(paragraphs) >= 2 and not paragraphs[1][0].upper() == 'B':
        score -= 0.15
    if len(paragraphs) >= 3 and not paragraphs[2][0].upper() == 'C':
        score -= 0.15

    # Word count per paragraph
    for p in paragraphs[:3]:
        if len(p.split()) > 50:
            score -= 0.1

    return max(0, score)


def score_word_count(response, task):
    """Check exact word count."""
    words = response.strip().split()
    target = task["target_words"]
    diff = abs(len(words) - target)
    if diff == 0:
        return 1.0
    elif diff <= 1:
        return 0.8
    elif diff <= 3:
        return 0.5
    elif diff <= 5:
        return 0.3
    return 0.0


def score_quality(response, task):
    """Basic quality scoring for subjective tasks."""
    score = 0.8  # Default decent score
    criteria = task.get("quality_criteria", [])
    text = response.strip()

    # Check basic structural criteria
    for c in criteria:
        if c == "exactly_2_sentences":
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if len(sentences) != 2:
                score -= 0.15
        elif c == "4_steps":
            if not re.search(r'[1234][\.\)]', text):
                score -= 0.1
        elif c == "numbered":
            if not re.search(r'[12345][\.\)]', text):
                score -= 0.1
        elif c == "single_sentence":
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            if len(sentences) > 1:
                score -= 0.15
        elif c == "under_30_words":
            if len(text.split()) > 30:
                score -= 0.1
        elif c == "all_5_sections" or c == "covers_all_5":
            nums_found = len(re.findall(r'[12345][\.\)]', text))
            if nums_found < 5:
                score -= 0.1
        elif c == "3_approaches":
            nums_found = len(re.findall(r'[123][\.\)]', text))
            if nums_found < 3:
                score -= 0.1

    # Basic length check (too short = probably bad)
    if len(text) < 50:
        score -= 0.2

    return max(0, min(1.0, score))


def score_structural_check(response, task):
    """Score based on objective structural validation."""
    checks = task.get("structural_checks", {})
    text = response.strip()
    score = 1.0
    deductions = 0

    # Count numbered items (e.g., "1." "2)" etc.)
    if "numbered_items" in checks:
        target = checks["numbered_items"]
        found = len(re.findall(r'(?:^|\n)\s*\d+[\.\):]', text))
        if found < target:
            deductions += 0.2 * (target - found) / target

    # Check required keywords (case-insensitive)
    if "required_keywords" in checks:
        keywords = checks["required_keywords"]
        missing = [k for k in keywords if not re.search(re.escape(k), text, re.IGNORECASE)]
        if missing:
            deductions += 0.15 * len(missing) / len(keywords)

    # Minimum character length
    if "min_length" in checks:
        if len(text) < checks["min_length"]:
            deductions += 0.2

    # Max word count
    if "max_words" in checks:
        word_count = len(text.split())
        if word_count > checks["max_words"]:
            deductions += 0.15

    # Max sentences
    if "max_sentences" in checks:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) > checks["max_sentences"]:
            deductions += 0.25

    # Exact sentence count
    if "exact_sentences" in checks:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) != checks["exact_sentences"]:
            deductions += 0.3

    return max(0.0, min(1.0, score - deductions))


SCORERS = {
    "exact_match": score_exact_match,
    "logic_check": score_exact_match,  # uses same regex pattern approach
    "constraint_check": score_exact_match,
    "code_execution": score_code_execution,
    "json_schema": score_json_schema,
    "instruction_check": score_instruction_check,
    "word_count": score_word_count,
    "quality": score_quality,
    "structural_check": score_structural_check,
}


# ─── Runner Functions ────────────────────────────────────────────────────────

def run_task_with_model(task, model_name, model_id):
    """Run a single task with a given model via claude CLI."""
    print(f"  [{model_name}] Running: {task['name']}...", end=" ", flush=True)

    start = time.time()
    try:
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", model_id,
                "--output-format", "json",
                "--max-budget-usd", "1.00",
                task["prompt"]
            ],
            capture_output=True, text=True, timeout=300
        )
        elapsed = time.time() - start

        output = result.stdout.strip()
        if not output:
            print(f"EMPTY (stderr: {result.stderr[:100]})")
            return {
                "task_id": task["id"],
                "model": model_name,
                "model_id": model_id,
                "response": "",
                "duration_ms": int(elapsed * 1000),
                "duration_api_ms": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "cost_usd": 0,
                "score": 0.0,
                "error": result.stderr[:200]
            }

        data = json.loads(output)
        response_text = data.get("result", "")

        # Score the response
        scorer = SCORERS.get(task["scoring"], score_quality)
        score = scorer(response_text, task)

        usage = data.get("usage", {})

        print(f"score={score:.1f}, {data.get('duration_ms', 0)}ms, ${data.get('total_cost_usd', 0):.4f}")

        return {
            "task_id": task["id"],
            "model": model_name,
            "model_id": model_id,
            "response": response_text,
            "duration_ms": data.get("duration_ms", int(elapsed * 1000)),
            "duration_api_ms": data.get("duration_api_ms", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
            "cost_usd": data.get("total_cost_usd", 0),
            "score": score,
            "error": None,
            "raw_json": data
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"TIMEOUT ({elapsed:.0f}s)")
        return {
            "task_id": task["id"],
            "model": model_name,
            "model_id": model_id,
            "response": "",
            "duration_ms": int(elapsed * 1000),
            "duration_api_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": 0,
            "score": 0.0,
            "error": "TIMEOUT"
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR: {e}")
        return {
            "task_id": task["id"],
            "model": model_name,
            "model_id": model_id,
            "response": "",
            "duration_ms": int(elapsed * 1000),
            "duration_api_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": 0,
            "score": 0.0,
            "error": str(e)
        }


def run_benchmark():
    """Run the full benchmark suite on both models."""
    models = [
        ("Opus 4.6", "claude-opus-4-6"),
        ("Opus 4.5", "claude-opus-4-5"),
    ]

    all_results = []

    for model_name, model_id in models:
        print(f"\n{'='*60}")
        print(f"  Running benchmark: {model_name} ({model_id})")
        print(f"{'='*60}")

        for task in TASKS:
            result = run_task_with_model(task, model_name, model_id)
            all_results.append(result)

    return all_results


def compute_summary(results):
    """Compute summary statistics from results."""
    models = {}
    categories = {}

    for r in results:
        model = r["model"]
        if model not in models:
            models[model] = {"scores": [], "latencies": [], "costs": [], "tokens_in": [], "tokens_out": []}

        models[model]["scores"].append(r["score"])
        models[model]["latencies"].append(r["duration_api_ms"])
        models[model]["costs"].append(r["cost_usd"])
        models[model]["tokens_in"].append(r["input_tokens"] + r["cache_read_tokens"] + r["cache_creation_tokens"])
        models[model]["tokens_out"].append(r["output_tokens"])

        # Find the task to get category
        task = next((t for t in TASKS if t["id"] == r["task_id"]), None)
        if task:
            cat = task["category"]
            key = f"{model}|{cat}"
            if key not in categories:
                categories[key] = {"scores": [], "model": model, "category": cat}
            categories[key]["scores"].append(r["score"])

    summary = {"models": {}, "categories": {}}

    for model, data in models.items():
        n = len(data["scores"])
        summary["models"][model] = {
            "avg_score": sum(data["scores"]) / n if n else 0,
            "avg_latency_ms": sum(data["latencies"]) / n if n else 0,
            "total_cost_usd": sum(data["costs"]),
            "avg_cost_usd": sum(data["costs"]) / n if n else 0,
            "avg_tokens_in": sum(data["tokens_in"]) / n if n else 0,
            "avg_tokens_out": sum(data["tokens_out"]) / n if n else 0,
            "total_tokens_in": sum(data["tokens_in"]),
            "total_tokens_out": sum(data["tokens_out"]),
            "task_count": n,
        }

    for key, data in categories.items():
        cat = data["category"]
        model = data["model"]
        if cat not in summary["categories"]:
            summary["categories"][cat] = {}
        n = len(data["scores"])
        summary["categories"][cat][model] = {
            "avg_score": sum(data["scores"]) / n if n else 0,
            "task_count": n,
        }

    return summary


def main():
    print("=" * 60)
    print("  Claude Opus 4.5 vs 4.6 Benchmark")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Tasks: {len(TASKS)}")
    print("=" * 60)

    results = run_benchmark()
    summary = compute_summary(results)

    # Build task metadata for the website
    task_meta = []
    for t in TASKS:
        task_meta.append({
            "id": t["id"],
            "category": t["category"],
            "name": t["name"],
            "prompt": t["prompt"],
            "expected_answer": t["expected_answer"],
            "scoring": t["scoring"],
        })

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": {
                "Opus 4.6": {"model_id": "claude-opus-4-6", "description": "Claude Opus 4.6 (latest)"},
                "Opus 4.5": {"model_id": "claude-opus-4-5", "description": "Claude Opus 4.5"},
            },
            "task_count": len(TASKS),
            "categories": list(set(t["category"] for t in TASKS)),
        },
        "summary": summary,
        "tasks": task_meta,
        "results": results,
    }

    output_path = "/home/ec2-user/opus-comparison/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Benchmark Complete!")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary table
    print(f"\n{'Model':<12} {'Avg Score':<12} {'Avg Latency':<14} {'Total Cost':<12}")
    print("-" * 50)
    for model, data in summary["models"].items():
        print(f"{model:<12} {data['avg_score']:.3f}        {data['avg_latency_ms']:.0f}ms         ${data['total_cost_usd']:.4f}")

    print(f"\nCategory Breakdown:")
    print(f"{'Category':<25} {'Opus 4.6':<12} {'Opus 4.5':<12}")
    print("-" * 50)
    for cat, model_scores in summary["categories"].items():
        s46 = model_scores.get("Opus 4.6", {}).get("avg_score", 0)
        s45 = model_scores.get("Opus 4.5", {}).get("avg_score", 0)
        print(f"{cat:<25} {s46:.3f}        {s45:.3f}")

    return output


if __name__ == "__main__":
    main()
