"""
Task definitions for the SQL Query Writing Environment.

Three tasks of increasing difficulty on a shared e-commerce SQLite database:
  Task 1 (easy)   — single-table filter + order
  Task 2 (medium) — multi-table JOIN + GROUP BY + aggregation
  Task 3 (hard)   — set intersection ("customers in BOTH months") + spend totals
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    email       TEXT,
    city        TEXT,
    country     TEXT,
    signup_date TEXT                -- YYYY-MM-DD
);

CREATE TABLE products (
    id       INTEGER PRIMARY KEY,
    name     TEXT    NOT NULL,
    category TEXT    NOT NULL,      -- 'Electronics' | 'Clothing' | 'Books' | 'Home'
    price    REAL    NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT    NOT NULL,   -- YYYY-MM-DD
    status      TEXT    NOT NULL    -- 'completed' | 'pending' | 'cancelled'
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price REAL    NOT NULL
);
"""

SEED_SQL = """
-- ── customers (15 rows) ────────────────────────────────────────────────────
INSERT INTO customers VALUES
  (1,  'Alice Martin',      'alice@example.com',   'Paris',         'France',        '2022-03-15'),
  (2,  'Bob Dupont',        'bob@example.com',     'Lyon',          'France',        '2021-08-20'),
  (3,  'Claire Leblanc',    'claire@example.com',  'Nice',          'France',        '2023-01-10'),
  (4,  'David Chen',        'david@example.com',   'Beijing',       'China',         '2022-06-01'),
  (5,  'Emma Schmidt',      'emma@example.com',    'Berlin',        'Germany',       '2021-11-30'),
  (6,  'Fatima Hassan',     'fatima@example.com',  'Cairo',         'Egypt',         '2023-05-15'),
  (7,  'George Brown',      'george@example.com',  'London',        'UK',            '2020-09-08'),
  (8,  'Helen Wilson',      'helen@example.com',   'London',        'UK',            '2022-04-22'),
  (9,  'Ivan Petrov',       'ivan@example.com',    'Moscow',        'Russia',        '2021-07-14'),
  (10, 'Julia Kim',         'julia@example.com',   'Seoul',         'South Korea',   '2022-12-01'),
  (11, 'Kevin Murphy',      'kevin@example.com',   'Dublin',        'Ireland',       '2023-03-17'),
  (12, 'Laura Santos',      'laura@example.com',   'Sao Paulo',     'Brazil',        '2022-09-25'),
  (13, 'Mike Johnson',      'mike@example.com',    'New York',      'USA',           '2021-05-03'),
  (14, 'Nancy Davis',       'nancy@example.com',   'Los Angeles',   'USA',           '2022-11-19'),
  (15, 'Oscar Rodriguez',   'oscar@example.com',   'Madrid',        'Spain',         '2023-07-04');

-- ── products (12 rows, 4 categories) ──────────────────────────────────────
INSERT INTO products VALUES
  (1,  'Laptop Pro',            'Electronics', 1299.99),
  (2,  'Smartphone X',          'Electronics',  799.99),
  (3,  'Wireless Headphones',   'Electronics',  199.99),
  (4,  'Smart Tablet',          'Electronics',  499.99),
  (5,  'Classic T-Shirt',       'Clothing',      29.99),
  (6,  'Slim Jeans',            'Clothing',      89.99),
  (7,  'Winter Jacket',         'Clothing',     179.99),
  (8,  'Summer Dress',          'Clothing',      99.99),
  (9,  'Python Programming',    'Books',         49.99),
  (10, 'ML Foundations',        'Books',         59.99),
  (11, 'Data Science Guide',    'Books',         39.99),
  (12, 'Premium Coffee Maker',  'Home',         129.99);

-- ── orders ────────────────────────────────────────────────────────────────
-- Jan 2024: customers 1,2,5,7,13 (all 5 will also order in Feb → Task 3)
--           + customer 4  (Jan only, NOT in Feb)
-- Feb 2024: same 5 + customer 8 (Feb only)
-- 2023:     general orders for Task 2 category revenue
INSERT INTO orders VALUES
  -- Jan 2024
  (1,  1,  '2024-01-10', 'completed'),
  (2,  2,  '2024-01-15', 'completed'),
  (3,  5,  '2024-01-20', 'completed'),
  (4,  7,  '2024-01-25', 'completed'),
  (5,  13, '2024-01-28', 'completed'),
  (6,  4,  '2024-01-30', 'completed'),   -- David: Jan only
  -- Feb 2024
  (7,  1,  '2024-02-05', 'completed'),
  (8,  2,  '2024-02-10', 'completed'),
  (9,  5,  '2024-02-14', 'completed'),
  (10, 7,  '2024-02-20', 'completed'),
  (11, 13, '2024-02-25', 'completed'),
  (12, 8,  '2024-02-28', 'completed'),   -- Helen: Feb only
  -- 2023 orders (for Task 2: category revenue)
  (13, 9,  '2023-03-15', 'completed'),
  (14, 10, '2023-04-20', 'completed'),
  (15, 11, '2023-05-10', 'completed'),
  (16, 12, '2023-06-18', 'completed'),
  (17, 14, '2023-07-22', 'completed'),
  (18, 15, '2023-08-14', 'completed'),
  (19, 6,  '2023-09-30', 'completed'),
  (20, 3,  '2023-10-12', 'completed'),
  (21, 6,  '2023-11-01', 'pending');     -- pending → excluded from revenue

-- ── order_items ───────────────────────────────────────────────────────────
INSERT INTO order_items VALUES
  -- Order 1  (Alice, Jan): Laptop Pro
  (1,  1,  1,  1, 1299.99),
  -- Order 2  (Bob, Jan): Smartphone + Headphones
  (2,  2,  2,  1,  799.99),
  (3,  2,  3,  1,  199.99),
  -- Order 3  (Emma, Jan): Winter Jacket + 2× T-Shirt
  (4,  3,  7,  1,  179.99),
  (5,  3,  5,  2,   29.99),
  -- Order 4  (George, Jan): 2× Python Programming
  (6,  4,  9,  2,   49.99),
  -- Order 5  (Mike, Jan): Laptop Pro + Coffee Maker
  (7,  5,  1,  1, 1299.99),
  (8,  5,  12, 1,  129.99),
  -- Order 6  (David, Jan): Smart Tablet
  (9,  6,  4,  1,  499.99),
  -- Order 7  (Alice, Feb): Headphones + T-Shirt
  (10, 7,  3,  1,  199.99),
  (11, 7,  5,  1,   29.99),
  -- Order 8  (Bob, Feb): Summer Dress
  (12, 8,  8,  1,   99.99),
  -- Order 9  (Emma, Feb): ML Foundations + Data Science Guide
  (13, 9,  10, 1,   59.99),
  (14, 9,  11, 1,   39.99),
  -- Order 10 (George, Feb): Smartphone X
  (15, 10, 2,  1,  799.99),
  -- Order 11 (Mike, Feb): Smart Tablet + Slim Jeans
  (16, 11, 4,  1,  499.99),
  (17, 11, 6,  1,   89.99),
  -- Order 12 (Helen, Feb): Coffee Maker + T-Shirt
  (18, 12, 12, 1,  129.99),
  (19, 12, 5,  1,   29.99),
  -- Order 13 (Ivan, 2023): Laptop Pro + Smartphone X
  (20, 13, 1,  1, 1299.99),
  (21, 13, 2,  1,  799.99),
  -- Order 14 (Julia, 2023): 3× T-Shirt + 2× Jeans + Winter Jacket
  (22, 14, 5,  3,   29.99),
  (23, 14, 6,  2,   89.99),
  (24, 14, 7,  1,  179.99),
  -- Order 15 (Kevin, 2023): 2× Python Programming + 3× ML Foundations
  (25, 15, 9,  2,   49.99),
  (26, 15, 10, 3,   59.99),
  -- Order 16 (Laura, 2023): 2× Coffee Maker + Laptop Pro
  (27, 16, 12, 2,  129.99),
  (28, 16, 1,  1, 1299.99),
  -- Order 17 (Nancy, 2023): Smart Tablet + 2× Headphones
  (29, 17, 4,  1,  499.99),
  (30, 17, 3,  2,  199.99),
  -- Order 18 (Oscar, 2023): Summer Dress + 2× T-Shirt + Slim Jeans
  (31, 18, 8,  1,   99.99),
  (32, 18, 5,  2,   29.99),
  (33, 18, 6,  1,   89.99),
  -- Order 19 (Fatima, 2023): 5× Data Science Guide
  (34, 19, 11, 5,   39.99),
  -- Order 20 (Claire, 2023): Wireless Headphones
  (35, 20, 3,  1,  199.99),
  -- Order 21 (Fatima, 2023): Laptop Pro — PENDING, should NOT appear in revenue
  (36, 21, 1,  1, 1299.99);
"""

# Human-readable schema shown to the agent in every observation
SCHEMA_DESCRIPTION = """\
=== DATABASE SCHEMA ===

TABLE: customers
  id          INTEGER  PRIMARY KEY
  name        TEXT
  email       TEXT
  city        TEXT
  country     TEXT
  signup_date TEXT  (format YYYY-MM-DD)

TABLE: products
  id       INTEGER  PRIMARY KEY
  name     TEXT
  category TEXT  (values: 'Electronics', 'Clothing', 'Books', 'Home')
  price    REAL

TABLE: orders
  id          INTEGER  PRIMARY KEY
  customer_id INTEGER  → customers.id
  order_date  TEXT     (format YYYY-MM-DD)
  status      TEXT     (values: 'completed', 'pending', 'cancelled')

TABLE: order_items
  id         INTEGER  PRIMARY KEY
  order_id   INTEGER  → orders.id
  product_id INTEGER  → products.id
  quantity   INTEGER
  unit_price REAL

=== SAMPLE ROWS ===

customers (15 total):
  (1, 'Alice Martin',   'Paris',       'France',      '2022-03-15')
  (2, 'Bob Dupont',     'Lyon',        'France',      '2021-08-20')
  (3, 'Claire Leblanc', 'Nice',        'France',      '2023-01-10')
  (4, 'David Chen',     'Beijing',     'China',       '2022-06-01')
  (5, 'Emma Schmidt',   'Berlin',      'Germany',     '2021-11-30')
  ... (10 more rows)

products (12 total):
  (1,  'Laptop Pro',           'Electronics', 1299.99)
  (2,  'Smartphone X',         'Electronics',  799.99)
  (5,  'Classic T-Shirt',      'Clothing',      29.99)
  (9,  'Python Programming',   'Books',         49.99)
  (12, 'Premium Coffee Maker', 'Home',         129.99)
  ... (7 more rows)

orders (21 total, mix of completed/pending):
  (1,  customer_id=1,  '2024-01-10', 'completed')
  (7,  customer_id=1,  '2024-02-05', 'completed')
  (13, customer_id=9,  '2023-03-15', 'completed')
  (21, customer_id=6,  '2023-11-01', 'pending')
  ... (17 more rows)

order_items (36 total):
  (1,  order_id=1,  product_id=1,  quantity=1, unit_price=1299.99)
  (2,  order_id=2,  product_id=2,  quantity=1, unit_price=799.99)
  ... (34 more rows)
"""

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    task_id: int
    difficulty: str
    question: str
    expected_sql: str
    max_attempts: int = 5


TASKS: dict[int, TaskDefinition] = {
    1: TaskDefinition(
        task_id=1,
        difficulty="easy",
        question=(
            "List the name and city of every customer from France, "
            "ordered alphabetically by name."
        ),
        expected_sql=(
            "SELECT name, city "
            "FROM customers "
            "WHERE country = 'France' "
            "ORDER BY name"
        ),
        max_attempts=5,
    ),
    2: TaskDefinition(
        task_id=2,
        difficulty="medium",
        question=(
            "What is the total revenue for each product category from completed orders? "
            "Return the category and total_revenue (sum of quantity × unit_price), "
            "ordered by total_revenue descending."
        ),
        expected_sql=(
            "SELECT p.category, "
            "       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.id "
            "JOIN orders   o ON oi.order_id   = o.id "
            "WHERE o.status = 'completed' "
            "GROUP BY p.category "
            "ORDER BY total_revenue DESC"
        ),
        max_attempts=5,
    ),
    3: TaskDefinition(
        task_id=3,
        difficulty="hard",
        question=(
            "Find customers who placed at least one completed order in January 2024 "
            "AND at least one completed order in February 2024. "
            "For each such customer return their name and total_spending "
            "(sum of quantity × unit_price across both months), "
            "ordered by total_spending descending."
        ),
        expected_sql=(
            "SELECT c.name, "
            "       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spending "
            "FROM customers   c "
            "JOIN orders      o  ON c.id        = o.customer_id "
            "JOIN order_items oi ON o.id         = oi.order_id "
            "WHERE o.status = 'completed' "
            "  AND o.order_date BETWEEN '2024-01-01' AND '2024-02-29' "
            "  AND c.id IN ( "
            "        SELECT customer_id FROM orders "
            "        WHERE  status = 'completed' AND order_date LIKE '2024-01-%' "
            "        INTERSECT "
            "        SELECT customer_id FROM orders "
            "        WHERE  status = 'completed' AND order_date LIKE '2024-02-%' "
            "  ) "
            "GROUP BY c.id, c.name "
            "ORDER BY total_spending DESC"
        ),
        max_attempts=5,
    ),
}

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def create_database() -> sqlite3.Connection:
    """Create a fresh in-memory SQLite database with schema + seed data."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL + SEED_SQL)
    conn.commit()
    return conn


def execute_query(conn: sqlite3.Connection, sql: str) -> tuple[list[dict], Optional[str]]:
    """
    Execute *sql* on *conn*.
    Returns (rows_as_dicts, None) on success or ([], error_message) on failure.
    """
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows, None
    except Exception as exc:
        return [], str(exc)


def compute_score(actual: list[dict], expected: list[dict]) -> float:
    """
    Jaccard similarity between two result sets.

    Each row is normalised to a frozenset of lower-cased string values so that:
    - column order doesn't matter
    - row order doesn't matter
    - minor float formatting differences are tolerated (values rounded to 2 dp)

    Returns a float in [0.0, 1.0].
    """
    def normalize(row: dict) -> frozenset:
        parts = set()
        for v in row.values():
            if isinstance(v, float):
                parts.add(f"{v:.2f}")
            else:
                parts.add(str(v).strip().lower())
        return frozenset(parts)

    actual_set   = {normalize(r) for r in actual}
    expected_set = {normalize(r) for r in expected}

    if not expected_set and not actual_set:
        return 1.0
    if not expected_set:
        return 0.0

    intersection = len(actual_set & expected_set)
    union        = len(actual_set | expected_set)
    return round(intersection / union, 4) if union > 0 else 0.0
