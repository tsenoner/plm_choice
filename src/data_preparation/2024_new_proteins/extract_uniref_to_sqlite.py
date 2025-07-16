#!/usr/bin/env python
"""
Parses UniRef XML files and directly loads cluster data into an SQLite database.
Merges functionality from extract_uniref_clusters.py and convert_to_efficient_format.py
to avoid intermediate JSON files.
"""

import gzip
import argparse
import json
import sqlite3
import re
from tqdm import tqdm
from pathlib import Path
import time

# Binary regex patterns for maximum speed (from extract_uniref_clusters.py)
ENTRY_PATTERN = re.compile(rb'<entry [^>]*id="([^"]+)"[^>]*>(.*?)</entry>', re.DOTALL)
UNIPARC_PATTERN = re.compile(
    rb'<dbReference[^>]+type="UniParc ID"[^>]+id="([^"]+)"[^>]*>', re.DOTALL
)
UNIPROTKB_PATTERN = re.compile(
    rb'<dbReference[^>]+type="UniProtKB ID"[^>]*>.*?<property[^>]+type="UniProtKB accession"[^>]+value="([^"]+)"[^>]*/>',
    re.DOTALL,
)


def extract_member_ids_from_content(entry_content_bytes):
    """
    Extracts member IDs from binary entry content.
    UniParc ID: extract id attribute directly.
    UniProtKB: extract value from UniProtKB accession property.
    Returns a list of unique member IDs (strings).
    """
    member_ids = []
    # Extract UniParc IDs
    uniparc_matches = UNIPARC_PATTERN.findall(entry_content_bytes)
    for match in uniparc_matches:
        member_ids.append(match.decode("utf-8"))

    # Extract UniProtKB accession values
    uniprotkb_matches = UNIPROTKB_PATTERN.findall(entry_content_bytes)
    for match in uniprotkb_matches:
        member_ids.append(match.decode("utf-8"))

    # Deduplicate while preserving order (important if order matters, otherwise set is fine)
    return list(dict.fromkeys(member_ids))


def parse_uniref_xml_to_sqlite(
    xml_file_path_str: str,
    sqlite_file_path_str: str,
    max_entries_to_process: int = None,
    xml_read_chunk_size: int = 1024 * 1024,  # 1MB
    sqlite_batch_size: int = 10000,
):
    """
    Parses a UniRef XML (.xml.gz) file and loads cluster data directly into an SQLite database.

    Args:
        xml_file_path_str (str): Path to the .xml.gz file.
        sqlite_file_path_str (str): Path to the output SQLite database file.
        max_entries_to_process (int, optional): Limit processing for testing.
        xml_read_chunk_size (int): Size of chunks to read from XML (bytes).
        sqlite_batch_size (int): Number of entries to batch before SQLite insertion.

    Returns:
        int: Number of cluster entries processed, or -1 on error.
    """
    entries_processed_count = 0
    sqlite_batch_data = []

    print(f"Processing XML file: {xml_file_path_str}")
    if max_entries_to_process is not None and max_entries_to_process > 0:
        print(f"Limiting processing to {max_entries_to_process} entries.")
    print(f"Outputting to SQLite database: {sqlite_file_path_str}")
    print(f"XML read chunk size: {xml_read_chunk_size / (1024**2):.1f} MB")
    print(f"SQLite batch insert size: {sqlite_batch_size:,} entries")

    conn = None
    try:
        conn = sqlite3.connect(sqlite_file_path_str)
        cursor = conn.cursor()

        # Create table - cluster_id as PRIMARY KEY automatically creates an index
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id TEXT PRIMARY KEY,
                members TEXT
            )
        """)
        conn.commit()

        progress_bar = tqdm(
            total=max_entries_to_process if max_entries_to_process else None,
            desc="UniRef XML to SQLite",
            unit=" entry",
            ncols=100,  # Wider progress bar
            unit_scale=True,
        )

        with gzip.open(xml_file_path_str, "rb") as f_xml:
            buffer = b""
            while True:
                if (
                    max_entries_to_process
                    and entries_processed_count >= max_entries_to_process
                ):
                    break

                chunk = f_xml.read(xml_read_chunk_size)
                if not chunk:
                    break
                buffer += chunk

                entry_matches = ENTRY_PATTERN.findall(buffer)

                for entry_match in entry_matches:
                    if (
                        max_entries_to_process
                        and entries_processed_count >= max_entries_to_process
                    ):
                        break

                    cluster_id_str = entry_match[0].decode("utf-8")
                    entry_content_bytes = entry_match[1]
                    member_ids = extract_member_ids_from_content(entry_content_bytes)

                    if member_ids:
                        members_json_str = json.dumps(member_ids)
                        sqlite_batch_data.append((cluster_id_str, members_json_str))
                        entries_processed_count += 1
                        progress_bar.update(1)

                        if len(sqlite_batch_data) >= sqlite_batch_size:
                            cursor.executemany(
                                "INSERT INTO clusters (cluster_id, members) VALUES (?, ?)",
                                sqlite_batch_data,
                            )
                            conn.commit()
                            sqlite_batch_data = []

                # Manage buffer: keep only the part after the last complete </entry>
                last_entry_end_pos = buffer.rfind(b"</entry>")
                if last_entry_end_pos != -1:
                    buffer = buffer[last_entry_end_pos + len(b"</entry>") :]
                elif (
                    not chunk and buffer
                ):  # No more chunks from file, but buffer has content not ending in </entry>
                    # This case implies malformed XML if entries are expected or just residual non-entry data
                    print(
                        f"Warning: Non-empty buffer remaining without a clear </entry> tag at EOF. Buffer size: {len(buffer)}"
                    )

            # Process any remaining entries in the buffer after EOF
            if buffer and not (
                max_entries_to_process
                and entries_processed_count >= max_entries_to_process
            ):
                entry_matches = ENTRY_PATTERN.findall(buffer)
                for entry_match in entry_matches:
                    if (
                        max_entries_to_process
                        and entries_processed_count >= max_entries_to_process
                    ):
                        break
                    cluster_id_str = entry_match[0].decode("utf-8")
                    entry_content_bytes = entry_match[1]
                    member_ids = extract_member_ids_from_content(entry_content_bytes)
                    if member_ids:
                        members_json_str = json.dumps(member_ids)
                        sqlite_batch_data.append((cluster_id_str, members_json_str))
                        entries_processed_count += 1
                        progress_bar.update(1)

            # Insert any final remaining batch
            if sqlite_batch_data:
                cursor.executemany(
                    "INSERT INTO clusters (cluster_id, members) VALUES (?, ?)",
                    sqlite_batch_data,
                )
                conn.commit()
                sqlite_batch_data = []

        progress_bar.close()
        print(
            f"Finished processing. {entries_processed_count} cluster entries written to SQLite."
        )
        return entries_processed_count

    except FileNotFoundError:
        print(f"Error: Input XML file not found at {xml_file_path_str}")
        return -1
    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
        return -1
    except IOError as e:
        print(f"I/O error: {e}")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Consider re-raising or logging traceback for debugging
        import traceback

        traceback.print_exc()
        return -1
    finally:
        if conn:
            conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Parse UniRef XML and load cluster data directly into an SQLite database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "xml_file", help="Path to the gzipped UniRef XML file (e.g., uniref50.xml.gz)."
    )
    parser.add_argument(
        "--output_sqlite",
        help="Optional path for the output SQLite database file. "
        "If not provided, output will be a .db file with the base name "
        "as the input XML file in the same directory (e.g., input.xml.gz -> input.db).",
        default=None,
    )
    parser.add_argument(
        "--process_limit",
        type=int,
        default=None,
        help="Limit processing to the first N entries (for testing).",
    )
    parser.add_argument(
        "--xml_read_chunk_size",
        type=int,
        default=1024 * 1024 * 10,  # 10MB
        help="Size of chunks to read from the XML file in bytes.",
    )
    parser.add_argument(
        "--sqlite_batch_size",
        type=int,
        default=100_000,
        help="Number of entries to accumulate before batch inserting into SQLite.",
    )

    args = parser.parse_args()
    input_xml_path = Path(args.xml_file)
    if args.output_sqlite:
        output_sqlite_path = Path(args.output_sqlite)
    else:
        # Derive output path (e.g., basename.xml.gz -> basename.db) by stripping .xml.gz and adding .db.
        base_path = input_xml_path.name.split(".", 1)[0]
        output_sqlite_path = input_xml_path.with_name(base_path + ".db")

    print(f"Input XML: {input_xml_path}")
    print(f"Output SQLite: {output_sqlite_path}")

    start_time = time.time()
    entries_count = parse_uniref_xml_to_sqlite(
        str(input_xml_path),
        str(output_sqlite_path),
        max_entries_to_process=args.process_limit,
        xml_read_chunk_size=args.xml_read_chunk_size,
        sqlite_batch_size=args.sqlite_batch_size,
    )
    end_time = time.time()

    if entries_count > 0:
        print(
            f"\nSuccessfully processed {entries_count} cluster entries to {output_sqlite_path}."
        )
        print(f"Total time: {end_time - start_time:.2f} seconds.")
        # Optionally, add info about the created DB (size, etc.)
        db_size_bytes = output_sqlite_path.stat().st_size
        print(f"SQLite database size: {db_size_bytes / (1024**3):.2f} GB")

    elif entries_count == 0:
        print(
            f"\nNo cluster entries found or processed. SQLite database at {output_sqlite_path} might be empty or only contain the table schema."
        )
    else:  # entries_count == -1
        print(
            f"\nAn error occurred. Output file {output_sqlite_path} may be incomplete or not correctly created."
        )
        print("Please check error messages above.")


if __name__ == "__main__":
    main()
