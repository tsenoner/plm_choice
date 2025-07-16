import sqlite3

DB_PATH = "data/2024_new_proteins/uniref50_2024_01.db"
TABLE_TO_DELETE = "all_2024_member_ids"


def delete_table():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print(f"Attempting to delete table '{TABLE_TO_DELETE}' from {DB_PATH}...")
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_TO_DELETE}")
        conn.commit()
        print(f"Table '{TABLE_TO_DELETE}' deleted successfully (if it existed).")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    delete_table()
