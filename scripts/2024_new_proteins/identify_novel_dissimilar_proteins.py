import json
import pandas as pd
import sqlite3

# Define file paths
NEW_PROTEINS_FILE = "data/explore/2024_newSeqs_pe12.tsv"
UNIREF50_2024_DB = "data/explore/uniref50_2024_01.db"
UNIREF50_2025_DB = "data/explore/uniref50_2025_01.db"
# OUTPUT_FILE = "data/explore/newly_discovered_dissimilar_proteins_2024.txt" # Old output
FINAL_PROTEINS_FASTA_FILE = (
    "data/explore/newly_discovered_dissimilar_proteins_2024.fasta"
)
SUMMARY_REPORT_FILE = "data/explore/analysis_summary_report.txt"

ALL_2024_MEMBERS_TABLE_NAME = "all_2024_member_ids"


def prepare_all_member_ids_table(db_path):
    """
    Ensures that a table named ALL_2024_MEMBERS_TABLE_NAME exists in the SQLite database
    at db_path and is populated with all unique member IDs from the 'clusters' table.
    """
    print(f"Preparing table '{ALL_2024_MEMBERS_TABLE_NAME}' in {db_path}...")
    conn = sqlite3.connect(db_path)

    # Initial cursor for table creation and count check
    init_cursor = conn.cursor()
    init_cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {ALL_2024_MEMBERS_TABLE_NAME} (
        member_id TEXT PRIMARY KEY
    )
    """)
    conn.commit()
    init_cursor.execute(f"SELECT COUNT(*) FROM {ALL_2024_MEMBERS_TABLE_NAME}")
    count = init_cursor.fetchone()[0]
    init_cursor.close()  # Close initial cursor

    if count > 0:
        print(
            f"Table '{ALL_2024_MEMBERS_TABLE_NAME}' already exists and is not empty ({count} rows). Skipping population."
        )
        conn.close()
        return

    print(f"Populating '{ALL_2024_MEMBERS_TABLE_NAME}' table. This may take a while...")

    # Use separate cursors for reading and writing
    read_cursor = conn.cursor()
    write_cursor = conn.cursor()

    read_cursor.execute("SELECT members FROM clusters")
    rows_processed = 0
    members_inserted = 0
    malformed_json_count = 0

    for row_index, row in enumerate(read_cursor):
        rows_processed += 1
        try:
            member_list = json.loads(row[0])
        except json.JSONDecodeError as e_json:
            malformed_json_count += 1
            if malformed_json_count <= 10:  # Log first few errors
                print(
                    f"Warning: JSONDecodeError for cluster at approx. row {row_index + 1}. Error: {e_json}. Data snippet: {row[0][:100]}"
                )
            if malformed_json_count == 11:
                print(
                    "Warning: Further JSONDecodeErrors will not be logged individually for performance."
                )
            continue  # Skip to next cluster if JSON is bad

        for member_id in member_list:
            try:
                write_cursor.execute(
                    f"INSERT OR IGNORE INTO {ALL_2024_MEMBERS_TABLE_NAME} (member_id) VALUES (?)",
                    (member_id,),
                )
                if write_cursor.rowcount > 0:  # check if INSERT actually happened
                    members_inserted += 1
            except sqlite3.InterfaceError as e_sqlite:
                # This might catch issues if member_id is not a string or is null, etc.
                print(
                    f"Skipping member ID due to sqlite3.InterfaceError: '{member_id}' (type: {type(member_id)}). Error: {e_sqlite}"
                )

        if rows_processed % 100000 == 0:
            conn.commit()  # Commit INSERTs periodically
            print(
                f"  Processed {rows_processed} clusters (approx. row {row_index + 1}), inserted {members_inserted} unique members so far. Malformed JSON rows: {malformed_json_count}"
            )

    conn.commit()  # Final commit for any remaining INSERTs

    print(f"Finished populating '{ALL_2024_MEMBERS_TABLE_NAME}'.")
    print(f"  Total clusters processed from 'clusters' table: {rows_processed}.")
    print(f"  Total unique member IDs inserted in this run: {members_inserted}.")
    print(
        f"  Total clusters with malformed JSON in 'members' column: {malformed_json_count}."
    )

    # Final count check
    count_cursor = conn.cursor()
    count_cursor.execute(f"SELECT COUNT(*) FROM {ALL_2024_MEMBERS_TABLE_NAME}")
    total_count_in_table = count_cursor.fetchone()[0]
    print(
        f"Total members in '{ALL_2024_MEMBERS_TABLE_NAME}' table after population: {total_count_in_table}."
    )

    read_cursor.close()
    write_cursor.close()
    count_cursor.close()
    conn.close()


def load_and_filter_new_proteins(file_path):
    """Loads new proteins, filters by PE level, returns dict {id: sequence}."""
    print(f"Loading and filtering new proteins from {file_path}...")
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["Entry", "Protein_existence", "Sequence"],
        engine="python",
    )
    filter_strings = ["Evidence at protein level", "Evidence at transcript level"]
    filtered_df = df[df["Protein_existence"].isin(filter_strings)]
    print(f"Found {len(filtered_df)} proteins with specified existence levels.")
    return dict(zip(filtered_df["Entry"], filtered_df["Sequence"]))


def get_cluster_ids_from_db(db_path):
    """Loads all UniRef50 cluster IDs from an SQLite database."""
    print(f"Loading UniRef50 cluster IDs from {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT cluster_id FROM clusters")
    cluster_ids = {row[0] for row in cursor.fetchall()}
    conn.close()
    print(f"Loaded {len(cluster_ids)} cluster IDs from {db_path}.")
    return cluster_ids


def get_members_of_truly_novel_clusters(
    candidate_new_cluster_ids, db_2025_path, db_2024_path
):
    """
    Identifies members of 2025 clusters that are "truly novel" and returns
    a set of all such members and a list of the truly novel cluster IDs.
    """
    print("Identifying members of truly novel 2025 clusters...")
    members_of_truly_novel_clusters_set = set()
    truly_novel_cluster_ids_list = []  # Store IDs of truly novel clusters

    conn_2025 = sqlite3.connect(db_2025_path)
    cursor_2025 = conn_2025.cursor()

    conn_2024 = sqlite3.connect(db_2024_path)
    cursor_2024 = conn_2024.cursor()

    candidate_ids_list = list(candidate_new_cluster_ids)
    total_candidates = len(candidate_ids_list)

    batch_size_2025_fetch = 1000

    for i in range(0, total_candidates, batch_size_2025_fetch):
        batch_candidate_ids = candidate_ids_list[i : i + batch_size_2025_fetch]
        placeholders = ", ".join(["?" for _ in batch_candidate_ids])
        query_2025 = f"SELECT cluster_id, members FROM clusters WHERE cluster_id IN ({placeholders})"
        cursor_2025.execute(query_2025, batch_candidate_ids)

        clusters_in_batch_processed = 0
        for cluster_id_2025, members_json_2025 in cursor_2025.fetchall():
            members_list_2025 = json.loads(members_json_2025)
            is_cluster_truly_novel = True
            if not members_list_2025:
                is_cluster_truly_novel = False

            for protein_id_2025 in members_list_2025:
                cursor_2024.execute(
                    f"SELECT 1 FROM {ALL_2024_MEMBERS_TABLE_NAME} WHERE member_id = ? LIMIT 1",
                    (protein_id_2025,),
                )
                if cursor_2024.fetchone():
                    is_cluster_truly_novel = False
                    break

            if is_cluster_truly_novel:
                members_of_truly_novel_clusters_set.update(members_list_2025)
                truly_novel_cluster_ids_list.append(cluster_id_2025)  # Add ID to list

            clusters_in_batch_processed += 1
            if (i + clusters_in_batch_processed) % 10000 == 0:
                print(
                    f"  Checked {i + clusters_in_batch_processed}/{total_candidates} candidate clusters. Found {len(truly_novel_cluster_ids_list)} truly novel clusters so far."
                )

    print(
        f"Finished checking. Found {len(truly_novel_cluster_ids_list)} truly novel 2025 clusters."
    )
    print(
        f"Total unique members in these truly novel clusters: {len(members_of_truly_novel_clusters_set)}."
    )

    conn_2025.close()
    conn_2024.close()
    return members_of_truly_novel_clusters_set, truly_novel_cluster_ids_list


def find_final_proteins(
    new_protein_ids_filtered_set, members_of_truly_novel_clusters_set
):
    """Finds the intersection between two sets of protein IDs."""
    print(
        "Finding intersection between new 2024 protein IDs and members of truly novel 2025 clusters..."
    )
    final_protein_ids_list = list(
        new_protein_ids_filtered_set.intersection(members_of_truly_novel_clusters_set)
    )
    print(f"Found {len(final_protein_ids_list)} proteins meeting all criteria.")
    return final_protein_ids_list


def save_final_proteins_to_fasta(
    final_protein_ids_list, all_protein_sequences_map, output_fasta_path
):
    """Saves a list of final protein IDs and their sequences to a FASTA file."""
    print(f"Saving final protein sequences to FASTA file: {output_fasta_path}...")
    with open(output_fasta_path, "w") as f:
        proteins_saved_count = 0
        for protein_id in final_protein_ids_list:
            sequence = all_protein_sequences_map.get(protein_id)
            if sequence:
                f.write(f">{protein_id}\n{sequence}\n")
                proteins_saved_count += 1
            else:
                print(
                    f"Warning: Sequence for protein ID '{protein_id}' not found in map. Skipping."
                )
    print(f"{proteins_saved_count} protein sequences saved to FASTA file.")


def save_summary_report(filepath, report_data):
    """Saves the analysis summary report to a text file."""
    print(f"Saving summary report to {filepath}...")
    with open(filepath, "w") as f:
        for key, value in report_data.items():
            f.write(f"{key}: {value}\n")
    print("Summary report saved.")


def main():
    prepare_all_member_ids_table(UNIREF50_2024_DB)

    new_proteins_2024_map = load_and_filter_new_proteins(NEW_PROTEINS_FILE)
    new_proteins_2024_filtered_ids = set(new_proteins_2024_map.keys())

    cluster_ids_2024 = get_cluster_ids_from_db(UNIREF50_2024_DB)
    cluster_ids_2025 = get_cluster_ids_from_db(UNIREF50_2025_DB)

    candidate_new_cluster_ids_2025 = cluster_ids_2025 - cluster_ids_2024
    num_candidate_new_cluster_ids = len(candidate_new_cluster_ids_2025)
    print(
        f"Found {num_candidate_new_cluster_ids} candidate new cluster IDs in 2025 DB."
    )

    members_of_truly_novel_2025_clusters_set, truly_novel_cluster_ids_list = (
        get_members_of_truly_novel_clusters(
            candidate_new_cluster_ids_2025, UNIREF50_2025_DB, UNIREF50_2024_DB
        )
    )
    num_truly_novel_cluster_ids = len(truly_novel_cluster_ids_list)

    # Calculate num_truly_novel_clusters_with_non_uniparc_members
    num_truly_novel_clusters_with_non_uniparc = 0
    if truly_novel_cluster_ids_list:
        print("Checking truly novel clusters for non-UniParc members...")
        conn_2025_temp = sqlite3.connect(UNIREF50_2025_DB)
        cursor_2025_temp = conn_2025_temp.cursor()
        processed_count = 0
        for cluster_id in truly_novel_cluster_ids_list:
            cursor_2025_temp.execute(
                "SELECT members FROM clusters WHERE cluster_id = ?", (cluster_id,)
            )
            result = cursor_2025_temp.fetchone()
            if result:
                members_json = result[0]
                member_list = json.loads(members_json)
                for member_id in member_list:
                    if not member_id.startswith("UPI"):
                        num_truly_novel_clusters_with_non_uniparc += 1
                        break
            processed_count += 1
            if processed_count % 1000 == 0:
                print(
                    f"  Checked {processed_count}/{num_truly_novel_cluster_ids} for non-UniParc members."
                )
        conn_2025_temp.close()
        print(
            f"Finished checking. Found {num_truly_novel_clusters_with_non_uniparc} truly novel clusters with non-UniParc (SwissProt/TrEMBL) members."
        )

    final_protein_ids_list = find_final_proteins(
        new_proteins_2024_filtered_ids, members_of_truly_novel_2025_clusters_set
    )
    num_final_proteins_found = len(final_protein_ids_list)

    # Calculate num_truly_novel_clusters_contributing_to_final_proteins
    num_truly_novel_clusters_contributing = 0
    if truly_novel_cluster_ids_list and final_protein_ids_list:
        print("Checking truly novel clusters for contribution to final proteins...")
        final_protein_ids_set = set(final_protein_ids_list)
        conn_2025_temp = sqlite3.connect(UNIREF50_2025_DB)
        cursor_2025_temp = conn_2025_temp.cursor()
        processed_count = 0
        for cluster_id in truly_novel_cluster_ids_list:
            cursor_2025_temp.execute(
                "SELECT members FROM clusters WHERE cluster_id = ?", (cluster_id,)
            )
            result = cursor_2025_temp.fetchone()
            if result:
                members_json = result[0]
                member_list = json.loads(members_json)
                for member_id in member_list:
                    if member_id in final_protein_ids_set:
                        num_truly_novel_clusters_contributing += 1
                        break
            processed_count += 1
            if processed_count % 1000 == 0:
                print(
                    f"  Checked {processed_count}/{num_truly_novel_cluster_ids} for contribution to final proteins."
                )
        conn_2025_temp.close()
        print(
            f"Finished checking. Found {num_truly_novel_clusters_contributing} truly novel clusters contributing to the final protein list."
        )

    report_data = {
        "Candidate new UniRef50 cluster IDs (in 2025 DB, not 2024 DB)": num_candidate_new_cluster_ids,
        "Truly novel UniRef50 cluster IDs (2025 clusters with no members from any 2024 cluster)": num_truly_novel_cluster_ids,
        "Truly novel 2025 clusters containing non-UniParc (SwissProt/TrEMBL) members": num_truly_novel_clusters_with_non_uniparc,
        "Final count of newly discovered proteins meeting all criteria (new in 2024, PE 1/2, in truly novel 2025 cluster)": num_final_proteins_found,
        "Number of truly novel 2025 clusters contributing at least one member to the final protein list": num_truly_novel_clusters_contributing,
    }

    save_summary_report(SUMMARY_REPORT_FILE, report_data)
    save_final_proteins_to_fasta(
        final_protein_ids_list, new_proteins_2024_map, FINAL_PROTEINS_FASTA_FILE
    )

    print("Analysis complete.")


if __name__ == "__main__":
    main()
