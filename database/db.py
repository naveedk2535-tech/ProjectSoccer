"""
Database connection and helper functions.
"""
import sqlite3
import os
import logging

import config

logger = logging.getLogger(__name__)


def get_connection():
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        schema = f.read()
    conn = get_connection()
    conn.executescript(schema)
    conn.close()
    logger.info("Database initialized at %s", config.DATABASE_PATH)


def execute(query, params=None):
    conn = get_connection()
    try:
        cursor = conn.execute(query, params or [])
        conn.commit()
        return cursor
    finally:
        conn.close()


def fetch_all(query, params=None):
    conn = get_connection()
    try:
        cursor = conn.execute(query, params or [])
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def fetch_one(query, params=None):
    conn = get_connection()
    try:
        cursor = conn.execute(query, params or [])
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def insert_many(table, rows):
    if not rows:
        return 0
    columns = rows[0].keys()
    placeholders = ", ".join(["?"] * len(columns))
    col_names = ", ".join(columns)
    query = f"INSERT OR IGNORE INTO {table} ({col_names}) VALUES ({placeholders})"
    conn = get_connection()
    try:
        cursor = conn.executemany(query, [tuple(r[c] for c in columns) for r in rows])
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def upsert(table, data, conflict_columns):
    columns = list(data.keys())
    placeholders = ", ".join(["?"] * len(columns))
    col_names = ", ".join(columns)
    conflict = ", ".join(conflict_columns)
    updates = ", ".join([f"{c}=excluded.{c}" for c in columns if c not in conflict_columns])
    query = f"""
        INSERT INTO {table} ({col_names}) VALUES ({placeholders})
        ON CONFLICT({conflict}) DO UPDATE SET {updates}
    """
    conn = get_connection()
    try:
        conn.execute(query, [data[c] for c in columns])
        conn.commit()
    finally:
        conn.close()
