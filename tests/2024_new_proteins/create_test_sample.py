import gzip
import re
from pathlib import Path


def create_sample_uniref(
    source_gz_file_path: str,
    output_xml_path: str,
    output_gz_path: str,
    num_entries: int = 10,
):
    """
    Extracts the first num_entries from a gzipped UniRef XML file
    and creates a new smaller gzipped XML file.
    """
    print(f"Creating sample with {num_entries} entries...")
    print(f"Source: {source_gz_file_path}")
    print(f"Output XML: {output_xml_path}")
    print(f"Output GZ: {output_gz_path}")

    entry_pattern = re.compile(rb"<entry.*?/entry>", re.DOTALL)
    entries_found = 0
    extracted_content = b""
    header_part = b""
    buffer = b""
    first_entry_tag_found_in_line = False

    try:
        f_in = None
        try:
            f_in = gzip.open(source_gz_file_path, "rb")
            # Try reading a byte to trigger BadGzipFile if it's not gzipped
            f_in.peek(1)
        except gzip.BadGzipFile:
            print(
                f"Warning: {source_gz_file_path} is not a gzipped file despite .gz extension. Opening as plain text."
            )
            f_in.close()  # Close the gzip handle if it was opened
            f_in = open(source_gz_file_path, "rb")  # Reopen as binary text
        except FileNotFoundError:
            raise  # Propagate FileNotFoundError
        except Exception as e_open:
            print(f"Error opening file {source_gz_file_path}: {e_open}")
            return

        with f_in:  # Manages closing for both gzip.open and open
            # Attempt to capture header more reliably
            for i, line in enumerate(f_in):
                header_part += line
                if b"<entry" in line:
                    # Split at the first occurrence of <entry
                    parts = line.split(b"<entry", 1)
                    header_part = header_part[
                        : -(len(line) - len(parts[0]))
                    ]  # Keep only content before <entry
                    buffer = (
                        b"<entry" + parts[1]
                    )  # Start buffer with the first entry tag and rest of the line
                    first_entry_tag_found_in_line = True
                    break
                if (
                    i > 200
                ):  # Safety break if no entry found in first 200 lines (typical header size)
                    print(
                        "Warning: No <entry> tag found within the first 200 lines. Proceeding with full read for entries."
                    )
                    break

            # If <entry> wasn't found quickly, the buffer might be empty, fill it.
            if not first_entry_tag_found_in_line:
                # Reset header_part if it became too large without finding an entry, or try to find a root tag
                if not header_part.strip().startswith(b"<uniprot"):
                    header_part = b""  # Will reconstruct later if needed
                buffer = f_in.read(
                    1024 * 1024 * 2
                )  # Read a larger initial chunk if header parsing was tricky

            # Ensure we have a root <uniprot ...> tag for the sample file
            if not header_part.strip().startswith(b"<uniprot"):
                # Try to find xmlns from the original source if possible (within the first few KB)
                # Open the source file again just to reliably read the header for xmlns
                initial_source_chunk = b""
                try:
                    # For header check, try gzip first, then plain open if it fails
                    temp_f_header_check = None
                    try:
                        temp_f_header_check = gzip.open(source_gz_file_path, "rb")
                        initial_source_chunk = temp_f_header_check.read(5 * 1024)
                        decompressed_initial_chunk = gzip.decompress(
                            initial_source_chunk
                        ).split(b"<entry", 1)[0]
                    except gzip.BadGzipFile:
                        if temp_f_header_check:
                            temp_f_header_check.close()
                        with open(source_gz_file_path, "rb") as plain_header_check:
                            decompressed_initial_chunk = plain_header_check.read(
                                5 * 1024
                            ).split(b"<entry", 1)[0]
                    finally:
                        if temp_f_header_check:
                            temp_f_header_check.close()

                    xmlns_match = re.search(
                        rb'<uniprot[^>]*xmlns="([^"]+)"[^>]*>',
                        decompressed_initial_chunk,
                        re.IGNORECASE,
                    )
                    if xmlns_match:
                        root_tag_match = re.search(
                            rb"(<uniprot[^>]*>)",
                            decompressed_initial_chunk,
                            re.IGNORECASE,
                        )
                        header = (
                            root_tag_match.group(1) + b"\n"
                            if root_tag_match
                            else b'<uniprot xmlns="http://uniprot.org/uniref">\n'
                        )
                    else:
                        header = b'<uniprot xmlns="http://uniprot.org/uniref">\n'  # Default if not found
                except Exception as e_header:
                    print(
                        f"Warning: Could not reliably read header for xmlns: {e_header}. Using default."
                    )
                    header = b'<uniprot xmlns="http://uniprot.org/uniref">\n'
            else:
                # Use the captured header, ensure it ends cleanly before entries
                header = header_part.split(b"<entry", 1)[0]
                if not header.strip().endswith(b">"):  # ensure it's a complete tag
                    header_lines = header.splitlines()
                    header = (
                        b"\n".join(header_lines[:-1]) + b"\n"
                        if len(header_lines) > 1
                        else b""
                    )
                    if not header.strip().startswith(b"<uniprot"):  # Failsafe
                        header = b'<uniprot xmlns="http://uniprot.org/uniref">\n'

            # Add entries
            while entries_found < num_entries:
                match = entry_pattern.search(buffer)
                if match:
                    entry_data = match.group(0)
                    extracted_content += entry_data + b"\n"
                    entries_found += 1
                    buffer = buffer[match.end() :]
                else:
                    chunk = f_in.read(1024 * 1024)  # Read 1MB
                    if not chunk:
                        print(
                            f"Warning: EOF reached but only {entries_found}/{num_entries} entries found."
                        )
                        break
                    buffer += chunk

            if entries_found < num_entries:
                print(
                    f"Warning: Could only extract {entries_found} out of {num_entries} requested entries."
                )

        final_xml_content = (
            header.strip() + b"\n" + extracted_content.strip() + b"\n</uniprot>\n"
        )

        Path(output_xml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_xml_path, "wb") as f_out_xml:
            f_out_xml.write(final_xml_content)
        print(f"Successfully wrote {entries_found} entries to {output_xml_path}")

        with gzip.open(output_gz_path, "wb") as f_out_gz:
            f_out_gz.write(final_xml_content)
        print(f"Successfully wrote gzipped sample to {output_gz_path}")

    except FileNotFoundError:
        print(f"Error: Source file {source_gz_file_path} not found.")
    except Exception as e:
        print(f"An error occurred during sample creation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Define paths relative to this script's location
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT_FROM_SAMPLE_CREATOR = (
        SCRIPT_DIR.parent.parent
    )  # tests/2024_new_proteins -> tests -> project_root

    source_file = (
        PROJECT_ROOT_FROM_SAMPLE_CREATOR / "data/explore/uniref50_2025_01.xml.gz"
    )
    sample_xml_dir = SCRIPT_DIR  # Output to the same directory as this script
    sample_xml = sample_xml_dir / "sample_uniref_10.xml"
    sample_gz = sample_xml_dir / "sample_uniref_10.xml.gz"

    # Ensure the target directory exists
    sample_xml_dir.mkdir(parents=True, exist_ok=True)

    create_sample_uniref(
        str(source_file), str(sample_xml), str(sample_gz), num_entries=10
    )
