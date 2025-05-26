#!/usr/bin/env python
"""
Final optimized UniRef cluster parser - closely follows the pattern of the super-fast entry ID extractor.
Focuses on the exact techniques that made the original so fast, with chunked writing and proper ID extraction.
"""

import gzip
import argparse
import json
from tqdm import tqdm
from pathlib import Path
import re


def parse_uniref_xml_clusters_final_optimized(
    xml_file_path,
    max_entries_to_process=None,
    output_json_filepath=None,
    chunk_size=1024 * 1024,  # 1MB chunks like the fast extractor
    write_chunk_size=1000,  # Write every 1000 entries to avoid RAM accumulation
):
    """
    Final optimized parser following the exact pattern of the fast entry ID extractor.
    Now with chunked writing to avoid RAM accumulation and proper ID extraction.

    Key optimizations from the fast entry ID extractor:
    - Binary regex processing to avoid encoding overhead
    - Large 1MB chunks for efficient I/O
    - Bulk regex operations with findall()
    - Chunked JSON writing to avoid memory accumulation
    - Stream processing to keep memory usage low
    - Proper ID extraction based on XML element types:
      * UniParc ID: extract id from <dbReference type="UniParc ID" id="UPI0035C05A72">
      * UniProtKB: extract value from <property type="UniProtKB accession" value="A0AAZ1XED0"/>
        when preceded by <dbReference type="UniProtKB ID" id="...">

    Args:
        xml_file_path (str): Path to the .xml.gz file
        max_entries_to_process (int, optional): Limit processing to first N entries
        output_json_filepath (str): Path to save extracted cluster data as JSON
        chunk_size (int): Size of chunks to read (default 1MB)
        write_chunk_size (int): Number of entries to accumulate before writing (default 1000)

    Returns:
        int: Number of cluster entries extracted, or -1 on error
    """
    entries_written_count = 0
    current_chunk_data = {}

    # Binary regex patterns for maximum speed
    entry_pattern = re.compile(
        rb'<entry [^>]*id="([^"]+)"[^>]*>(.*?)</entry>', re.DOTALL
    )

    # Pattern to find UniParc ID dbReferences
    uniparc_pattern = re.compile(
        rb'<dbReference[^>]+type="UniParc ID"[^>]+id="([^"]+)"[^>]*>', re.DOTALL
    )

    # Pattern to find UniProtKB ID dbReferences followed by UniProtKB accession properties
    uniprotkb_pattern = re.compile(
        rb'<dbReference[^>]+type="UniProtKB ID"[^>]*>.*?<property[^>]+type="UniProtKB accession"[^>]+value="([^"]+)"[^>]*/>',
        re.DOTALL,
    )

    print(f"Processing XML file: {xml_file_path}")
    if max_entries_to_process is not None and max_entries_to_process > 0:
        print(f"Limiting processing to {max_entries_to_process} entries.")
    print(f"Extracting cluster data to JSON file: {output_json_filepath}")
    print(f"Writing in chunks of {write_chunk_size} entries to manage memory usage.")

    def extract_member_ids(entry_content):
        """
        Extract member IDs from entry content based on XML element types:
        - UniParc ID: extract id attribute directly
        - UniProtKB: extract value from UniProtKB accession property
        """
        member_ids = []

        # Extract UniParc IDs
        uniparc_matches = uniparc_pattern.findall(entry_content)
        for match in uniparc_matches:
            member_ids.append(match.decode("utf-8"))

        # Extract UniProtKB accession values
        uniprotkb_matches = uniprotkb_pattern.findall(entry_content)
        for match in uniprotkb_matches:
            member_ids.append(match.decode("utf-8"))

        # Deduplicate while preserving order
        return list(dict.fromkeys(member_ids))

    def write_chunk_to_file(outfile, chunk_data, is_first_chunk, is_last_chunk):
        """Write a chunk of data to the JSON file with proper formatting"""
        if not chunk_data:
            return

        for i, (cluster_id, member_ids) in enumerate(chunk_data.items()):
            # Add comma separator if not the first entry overall
            if not (is_first_chunk and i == 0):
                outfile.write(",\n  ")
            elif is_first_chunk and i == 0:
                outfile.write("\n  ")

            # Write the entry
            outfile.write(f'"{cluster_id}": {json.dumps(member_ids)}')

        # Flush the buffer to ensure data is written
        outfile.flush()

    try:
        progress_bar = tqdm(
            total=max_entries_to_process if max_entries_to_process else None,
            desc="Extracting UniRef clusters",
            unit=" entry",
            ncols=80,
            unit_scale=True,
        )

        with open(output_json_filepath, "w", buffering=8192) as outfile:
            # Start JSON object
            outfile.write("{")
            is_first_chunk = True

            with gzip.open(xml_file_path, "rb") as f_xml:
                buffer = b""

                while True:
                    if (
                        max_entries_to_process
                        and entries_written_count >= max_entries_to_process
                    ):
                        break

                    # Read large chunk - exactly like the fast extractor
                    chunk = f_xml.read(chunk_size)
                    if not chunk:
                        break

                    # Add to buffer
                    buffer += chunk

                    # Find all complete entries in the current buffer
                    entry_matches = entry_pattern.findall(buffer)

                    for entry_match in entry_matches:
                        if (
                            max_entries_to_process
                            and entries_written_count >= max_entries_to_process
                        ):
                            break

                        cluster_id = entry_match[0].decode("utf-8")
                        entry_content = entry_match[1]

                        # Extract member IDs based on XML element types
                        member_ids = extract_member_ids(entry_content)

                        if member_ids:
                            current_chunk_data[cluster_id] = member_ids
                            entries_written_count += 1
                            progress_bar.update(1)

                            # Write chunk when we reach the chunk size
                            if len(current_chunk_data) >= write_chunk_size:
                                write_chunk_to_file(
                                    outfile, current_chunk_data, is_first_chunk, False
                                )
                                current_chunk_data = {}
                                is_first_chunk = False

                    # Keep only the last incomplete part in buffer - exactly like fast extractor
                    last_newline = buffer.rfind(b"</entry>")
                    if last_newline != -1:
                        buffer = buffer[last_newline + 8 :]  # 8 = len("</entry>")

                    # Break if we've hit our limit
                    if (
                        max_entries_to_process
                        and entries_written_count >= max_entries_to_process
                    ):
                        break

                # Process remaining buffer - exactly like fast extractor
                if buffer and not (
                    max_entries_to_process
                    and entries_written_count >= max_entries_to_process
                ):
                    entry_matches = entry_pattern.findall(buffer)
                    for entry_match in entry_matches:
                        if (
                            max_entries_to_process
                            and entries_written_count >= max_entries_to_process
                        ):
                            break

                        cluster_id = entry_match[0].decode("utf-8")
                        entry_content = entry_match[1]

                        member_ids = extract_member_ids(entry_content)

                        if member_ids:
                            current_chunk_data[cluster_id] = member_ids
                            entries_written_count += 1
                            progress_bar.update(1)

            # Write any remaining data in the final chunk
            if current_chunk_data:
                write_chunk_to_file(outfile, current_chunk_data, is_first_chunk, True)

            # Close JSON object
            outfile.write("\n}")

        progress_bar.close()

        print(
            f"Finished processing XML. {entries_written_count} cluster entries extracted and written."
        )
        return entries_written_count

    except FileNotFoundError:
        print(f"Error: Input XML file not found at {xml_file_path}")
        return -1
    except IOError as e:
        print(
            f"Error: Could not open or write to output file {output_json_filepath}. Error: {e}"
        )
        return -1
    except Exception as e:
        print(f"An unexpected error occurred during parsing or writing: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser(
        description="Final optimized UniRef cluster parser with chunked writing and proper ID extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "xml_file", help="Path to the gzipped UniRef XML file (e.g., uniref50.xml.gz)."
    )
    parser.add_argument(
        "--output_json",
        help="Optional path to save the extracted cluster data as a JSON file. "
        "If not provided, output will be saved to a .json file with the base name "
        "as the input XML file in the same directory "
        "(e.g., input.xml.gz -> input.json).",
        default=None,
    )
    parser.add_argument(
        "--process_limit",
        type=int,
        default=None,
        help="Limit processing to the first N entries for testing.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 1024,
        help="Size of chunks to read in bytes (default: 1MB).",
    )
    parser.add_argument(
        "--write_chunk_size",
        type=int,
        default=1000,
        help="Number of entries to accumulate before writing to file (default: 1000).",
    )

    args = parser.parse_args()

    input_xml_path = Path(args.xml_file)
    if args.output_json:
        output_json_path = Path(args.output_json)
    else:
        base_name = input_xml_path.stem
        if base_name.endswith(".xml"):
            base_name = Path(base_name).stem
        output_json_path = input_xml_path.with_name(base_name).with_suffix(".json")

    print(f"Output will be saved to: {output_json_path}")
    print(f"Write chunk size: {args.write_chunk_size} entries")

    entries_written_count = parse_uniref_xml_clusters_final_optimized(
        args.xml_file,
        args.process_limit,
        output_json_filepath=str(output_json_path),
        chunk_size=args.chunk_size,
        write_chunk_size=args.write_chunk_size,
    )

    if entries_written_count > 0:
        print(
            f"\nSuccessfully processed {entries_written_count} cluster entries to {output_json_path}."
        )
        print(f"JSON file completed and saved to {output_json_path}")
    elif entries_written_count == 0:
        print(
            f"\nNo cluster entries found or processed. Empty JSON object saved to {output_json_path}."
        )
    else:  # entries_written_count == -1
        print(
            f"\nAn error occurred during processing. Output file {output_json_path} may be incomplete or not created."
        )
        print("Please check previous error messages.")


if __name__ == "__main__":
    main()
