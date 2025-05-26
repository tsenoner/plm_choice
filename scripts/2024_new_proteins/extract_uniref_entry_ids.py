#!/usr/bin/env python
import gzip
import argparse
import json
from tqdm import tqdm
from pathlib import Path
import re


def parse_uniref_xml_entry_ids(
    xml_file_path,
    max_entries_to_process=None,
    output_json_filepath=None,
    chunk_size=1024 * 1024,  # 1MB chunks for maximum I/O performance
):
    """
    High-performance parser for extracting UniRef entry IDs from gzipped XML files.

    Optimizations:
    - Large 1MB chunks for efficient I/O
    - Binary processing to avoid encoding overhead
    - Bulk regex operations with findall()
    - Single JSON write at the end
    - Stream processing to keep memory usage low

    Args:
        xml_file_path (str): Path to the .xml.gz file
        max_entries_to_process (int, optional): Limit processing to first N entries
        output_json_filepath (str): Path to save extracted entry IDs as JSON list
        chunk_size (int): Size of chunks to read (default 1MB)

    Returns:
        int: Number of entry IDs extracted, or -1 on error
    """
    ids_extracted = []
    ids_written_to_json_count = 0

    # Binary regex pattern for maximum speed
    entry_pattern = re.compile(rb'<entry [^>]*id="([^"]+)"[^>]*>')

    print(f"Processing XML file: {xml_file_path}")
    if max_entries_to_process is not None and max_entries_to_process > 0:
        print(f"Limiting processing to {max_entries_to_process} <entry> elements.")
    print(f"Extracting entry IDs to JSON file: {output_json_filepath}")

    try:
        progress_bar = tqdm(
            total=max_entries_to_process if max_entries_to_process else None,
            desc="Extracting UniRef entry IDs",
            unit=" entry",
            ncols=80,
            unit_scale=True,
        )

        with gzip.open(xml_file_path, "rb") as f_xml:
            buffer = b""

            while True:
                if (
                    max_entries_to_process
                    and ids_written_to_json_count >= max_entries_to_process
                ):
                    break

                # Read large chunk
                chunk = f_xml.read(chunk_size)
                if not chunk:
                    break

                # Add to buffer
                buffer += chunk

                # Find all matches in the current buffer
                matches = entry_pattern.findall(buffer)

                for match in matches:
                    if (
                        max_entries_to_process
                        and ids_written_to_json_count >= max_entries_to_process
                    ):
                        break

                    entry_id = match.decode("utf-8")
                    ids_extracted.append(entry_id)
                    ids_written_to_json_count += 1
                    progress_bar.update(1)

                # Keep only the last incomplete line in buffer
                last_newline = buffer.rfind(b"\n")
                if last_newline != -1:
                    buffer = buffer[last_newline + 1 :]

                # Break if we've hit our limit
                if (
                    max_entries_to_process
                    and ids_written_to_json_count >= max_entries_to_process
                ):
                    break

            # Process remaining buffer
            if buffer and not (
                max_entries_to_process
                and ids_written_to_json_count >= max_entries_to_process
            ):
                matches = entry_pattern.findall(buffer)
                for match in matches:
                    if (
                        max_entries_to_process
                        and ids_written_to_json_count >= max_entries_to_process
                    ):
                        break
                    entry_id = match.decode("utf-8")
                    ids_extracted.append(entry_id)
                    ids_written_to_json_count += 1
                    progress_bar.update(1)

        progress_bar.close()

        # Write all IDs to JSON file at once
        print("Writing extracted IDs to JSON file...")
        with open(output_json_filepath, "w") as outfile:
            json.dump(ids_extracted, outfile, indent=2)

        print(
            f"Finished processing XML. {ids_written_to_json_count} entry IDs extracted and written."
        )
        return ids_written_to_json_count

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
        description="High-performance extractor for UniRef entry IDs from UniRef XML data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "xml_file", help="Path to the gzipped UniRef XML file (e.g., uniref50.xml.gz)."
    )
    parser.add_argument(
        "--output_json",
        help="Optional path to save the extracted entry IDs as a JSON list file. "
        "If not provided, output will be saved to a .json file with the base name "
        "as the input XML file plus '_entry_ids' suffix in the same directory "
        "(e.g., input.xml.gz -> input_entry_ids.json).",
        default=None,
    )
    parser.add_argument(
        "--process_limit",
        type=int,
        default=None,
        help="Limit processing to the first N <entry> elements for testing.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 1024,
        help="Size of chunks to read in bytes (default: 1MB).",
    )

    args = parser.parse_args()

    input_xml_path = Path(args.xml_file)
    if args.output_json:
        output_json_path = Path(args.output_json)
    else:
        base_name = input_xml_path.stem
        if base_name.endswith(".xml"):
            base_name = Path(base_name).stem
        output_json_path = input_xml_path.with_name(
            f"{base_name}_entry_ids"
        ).with_suffix(".json")

    print(f"Output will be saved to: {output_json_path}")

    ids_written_count = parse_uniref_xml_entry_ids(
        args.xml_file,
        args.process_limit,
        output_json_filepath=str(output_json_path),
        chunk_size=args.chunk_size,
    )

    if ids_written_count > 0:
        print(
            f"\nSuccessfully processed {ids_written_count} entry IDs to {output_json_path}."
        )
        print(f"JSON file completed and saved to {output_json_path}")
    elif ids_written_count == 0:
        print(
            f"\nNo entry IDs found or processed. Empty JSON list saved to {output_json_path}."
        )
    else:  # ids_written_count == -1
        print(
            f"\nAn error occurred during processing. Output file {output_json_path} may be incomplete or not created."
        )
        print("Please check previous error messages.")


if __name__ == "__main__":
    main()
