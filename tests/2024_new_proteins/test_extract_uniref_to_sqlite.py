import pytest
import subprocess
import sqlite3
import json
from pathlib import Path

# --- Base Directory Setup ---
THIS_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = (
    THIS_FILE_DIR.parent.parent
)  # Assumes tests/<feature_subdir>/this_file.py

# --- Configurable Path Components ---
_FEATURE_NAME_PART = "2024_new_proteins"
_SAMPLE_BASENAME = "sample_uniref_10"
_SCRIPT_TO_TEST_FILENAME = "extract_uniref_to_sqlite.py"
_SAMPLE_CREATOR_FILENAME = "create_test_sample.py"
_SOURCE_DATA_INPUT_FILENAME = "uniref50_2025_01.xml.gz"

# --- Derived Paths ---
TEST_DIR = THIS_FILE_DIR
SCRIPT_TO_TEST = (
    PROJECT_ROOT / "scripts" / _FEATURE_NAME_PART / _SCRIPT_TO_TEST_FILENAME
)
SAMPLE_CREATOR_SCRIPT = TEST_DIR / _SAMPLE_CREATOR_FILENAME
SOURCE_DATA_FOR_SAMPLE = (
    PROJECT_ROOT / "data" / _FEATURE_NAME_PART / _SOURCE_DATA_INPUT_FILENAME
)
SAMPLE_XML_GZ = TEST_DIR / f"{_SAMPLE_BASENAME}.xml.gz"
SAMPLE_XML = TEST_DIR / f"{_SAMPLE_BASENAME}.xml"  # May be used by other logic
SAMPLE_DB = TEST_DIR / f"{_SAMPLE_BASENAME}.db"

EXPECTED_DATA = [
    {"cluster_id": "UniRef50_UPI002E2621C6", "members": ["UPI002E2621C6"]},
    {"cluster_id": "UniRef50_UPI00358F51CD", "members": ["UPI00358F51CD"]},
    {"cluster_id": "UniRef50_A0A5A9P0L4", "members": ["A0A5A9P0L4"]},
    {"cluster_id": "UniRef50_A0AB34IYJ6", "members": ["A0AB34IYJ6"]},
    {"cluster_id": "UniRef50_UPI00312B5ECC", "members": ["UPI00312B5ECC"]},
    {"cluster_id": "UniRef50_UPI0016133188", "members": ["UPI0016133188"]},
    {"cluster_id": "UniRef50_UPI002E22A622", "members": ["UPI002E22A622"]},
    {"cluster_id": "UniRef50_A0A410P257", "members": ["A0A410P257", "UPI000FFEDAD7"]},
    {"cluster_id": "UniRef50_A0A8J3NBY6", "members": ["A0A8J3NBY6", "UPI0019403D63"]},
    {"cluster_id": "UniRef50_Q8WZ42", "exact_member_count": 4446},
]


@pytest.fixture(scope="session")
def sample_xml_setup():
    """Session-scoped fixture to create the sample XML.gz file once."""
    print("Setting up test environment (session-scoped sample XML)...")
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_CREATOR_SCRIPT.exists():
        raise FileNotFoundError(
            f"Sample creator script not found: {SAMPLE_CREATOR_SCRIPT}"
        )
    if not SOURCE_DATA_FOR_SAMPLE.exists():
        raise FileNotFoundError(
            f"Source data for sample creation not found: {SOURCE_DATA_FOR_SAMPLE}"
        )

    print(f"Running sample creator: {SAMPLE_CREATOR_SCRIPT}")
    process = subprocess.run(
        [
            "python",
            str(SAMPLE_CREATOR_SCRIPT),
            str(SOURCE_DATA_FOR_SAMPLE),
            str(SAMPLE_XML_GZ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        print("Sample creator STDOUT:", process.stdout)
        print("Sample creator STDERR:", process.stderr)
        raise AssertionError("Sample XML creation failed.")
    if not SAMPLE_XML_GZ.exists():
        raise FileNotFoundError(f"Sample XML.gz file not created: {SAMPLE_XML_GZ}")
    print("Sample XML.gz created successfully.")

    yield

    print("Cleaning up test environment (session-scoped sample XML)...")
    if SAMPLE_XML_GZ.exists():
        SAMPLE_XML_GZ.unlink()
        print(f"Cleaned up {SAMPLE_XML_GZ}")
    if SAMPLE_XML.exists():
        SAMPLE_XML.unlink()
        print(f"Cleaned up {SAMPLE_XML}")


@pytest.fixture(scope="function")
def sample_db_setup(sample_xml_setup):
    """Function-scoped fixture to run the script and create the SQLite DB for each test."""
    if SAMPLE_DB.exists():
        SAMPLE_DB.unlink()

    if not SCRIPT_TO_TEST.exists():
        raise FileNotFoundError(f"Script to test not found: {SCRIPT_TO_TEST}")

    cmd = [
        "python",
        str(SCRIPT_TO_TEST),
        str(SAMPLE_XML_GZ),
        "--output_sqlite",
        str(SAMPLE_DB),
    ]
    cmd_str = " ".join(cmd)
    print(f"Running script: {cmd_str}")
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        print("Script STDOUT:", process.stdout)
        print("Script STDERR:", process.stderr)
        raise AssertionError(f"Script {SCRIPT_TO_TEST} execution failed.")
    if not SAMPLE_DB.exists():
        raise FileNotFoundError(f"SQLite DB not created by script: {SAMPLE_DB}")
    print(f"SQLite DB {SAMPLE_DB} created successfully for test.")

    yield SAMPLE_DB

    print(f"Cleaning up {SAMPLE_DB} after test.")
    if SAMPLE_DB.exists():
        SAMPLE_DB.unlink()


def test_database_creation_and_table_schema(sample_db_setup):
    """Test if the database and clusters table are created with correct schema."""
    db_path = sample_db_setup
    assert db_path.exists(), "SQLite database file was not created."
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='clusters';"
    )
    assert cursor.fetchone() is not None, "'clusters' table not found."

    cursor.execute("PRAGMA table_info(clusters);")
    columns_info = {row[1]: row[2] for row in cursor.fetchall()}
    assert "cluster_id" in columns_info, "'cluster_id' column missing."
    assert columns_info["cluster_id"].upper() == "TEXT", (
        "'cluster_id' column type is not TEXT."
    )

    # Check for PRIMARY KEY on cluster_id
    # Need to re-execute PRAGMA or store its full result as fetchall exhausts the cursor.
    cursor.execute("PRAGMA table_info(clusters);")
    pk_column_info = next(
        (row for row in cursor.fetchall() if row[1] == "cluster_id"), None
    )
    assert pk_column_info is not None, "Could not get info for cluster_id for PK check"
    assert pk_column_info[5] == 1, "'cluster_id' is not PRIMARY KEY"

    assert "members" in columns_info, "'members' column missing."
    assert columns_info["members"].upper() == "TEXT", (
        "'members' column type is not TEXT."
    )
    conn.close()


def test_cluster_count(sample_db_setup):
    """Test if the correct number of clusters (10 for the sample) are inserted."""
    db_path = sample_db_setup
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM clusters;")
    count = cursor.fetchone()[0]
    conn.close()
    assert count == 10, "Incorrect number of clusters in the database."


def test_specific_cluster_data(sample_db_setup):
    """Test parsing of specific cluster IDs and their members."""
    db_path = sample_db_setup
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for expected in EXPECTED_DATA:
        cursor.execute(
            "SELECT members FROM clusters WHERE cluster_id = ?;",
            (expected["cluster_id"],),
        )
        row = cursor.fetchone()
        assert row is not None, f"Cluster ID {expected['cluster_id']} not found."
        actual_members = json.loads(row[0])

        if "exact_member_count" in expected:
            assert len(actual_members) == expected["exact_member_count"], (
                f"Cluster {expected['cluster_id']} should have exactly "
                f"{expected['exact_member_count']} members. Found {len(actual_members)}."
            )
        elif "members" in expected:
            # Script preserves member order; direct comparison if order is guaranteed.
            # Otherwise, use sorted comparison.
            assert sorted(actual_members) == sorted(expected["members"]), (
                f"Member list mismatch for cluster {expected['cluster_id']}. "
                f"Expected {sorted(expected['members'])}, got {sorted(actual_members)}"
            )
    conn.close()
