import gzip
import re
from pathlib import Path
import argparse


def create_sample_uniref(
    source_gz_file_path: str,
    output_gz_path: str,
    num_entries: int = 10,
):
    """Extracts n entries from a gzipped UniRef XML to a new gzipped file."""
    print(
        f"Creating sample: {num_entries} entries from {source_gz_file_path} to {output_gz_path}"
    )

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
            f_in.peek(1)  # Check if gzipped
        except gzip.BadGzipFile:
            print(f"Warning: {source_gz_file_path} not gzipped. Opening as plain text.")
            if f_in:
                f_in.close()
            f_in = open(source_gz_file_path, "rb")
        except FileNotFoundError:
            raise
        except Exception as e_open:
            print(f"Error opening {source_gz_file_path}: {e_open}")
            return

        with f_in:
            # Capture header before the first <entry>
            for i, line in enumerate(f_in):
                header_part += line
                if b"<entry" in line:
                    parts = line.split(b"<entry", 1)
                    header_part = header_part[: -(len(line) - len(parts[0]))]
                    buffer = b"<entry" + parts[1]
                    first_entry_tag_found_in_line = True
                    break
                if i > 200:  # Safety break
                    print("Warning: No <entry> in first 200 lines.")
                    break

            if not first_entry_tag_found_in_line:
                if not header_part.strip().startswith(b"<uniprot"):
                    header_part = b""
                buffer = f_in.read(1024 * 1024 * 2)  # Read 2MB chunk

            # Ensure a root <uniprot ...> tag
            if not header_part.strip().startswith(b"<uniprot"):
                initial_source_chunk = b""
                try:
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
                        header = b'<uniprot xmlns="http://uniprot.org/uniref">\n'
                except Exception as e_header:
                    print(
                        f"Warning: Could not read header for xmlns: {e_header}. Using default."
                    )
                    header = b'<uniprot xmlns="http://uniprot.org/uniref">\n'
            else:
                header = header_part.split(b"<entry", 1)[0]
                if not header.strip().endswith(b">"):
                    header_lines = header.splitlines()
                    header = (
                        b"\n".join(header_lines[:-1]) + b"\n"
                        if len(header_lines) > 1
                        else b""
                    )
                    if not header.strip().startswith(b"<uniprot"):
                        header = b'<uniprot xmlns="http://uniprot.org/uniref">\n'  # Default bytes

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
                            f"Warning: EOF, {entries_found}/{num_entries} entries found."
                        )
                        break
                    buffer += chunk

            if entries_found < num_entries:
                print(f"Warning: Extracted {entries_found}/{num_entries} entries.")

        final_xml_content = (
            header.strip() + b"\n" + extracted_content.strip() + b"\n</uniprot>\n"
        )

        Path(output_gz_path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(output_gz_path, "wb") as f_out_gz:
            f_out_gz.write(final_xml_content)
        print(f"Successfully wrote gzipped sample to {output_gz_path}")

    except FileNotFoundError:
        print(f"Error: Source file {source_gz_file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a sample UniRef gzipped XML file."
    )
    parser.add_argument(
        "source_file_path", type=str, help="Source gzipped UniRef XML file."
    )
    parser.add_argument(
        "output_gz_file_path", type=str, help="Output sample gzipped XML file."
    )
    parser.add_argument(
        "--num_entries", type=int, default=10, help="Number of entries (default: 10)."
    )
    args = parser.parse_args()

    create_sample_uniref(
        args.source_file_path,
        args.output_gz_file_path,
        num_entries=args.num_entries,
    )
