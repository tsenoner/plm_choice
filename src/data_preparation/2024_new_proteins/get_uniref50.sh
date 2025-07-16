#!/bin/bash

set -e # Exit on error

# === Usage Check ===
if [ $# -ne 1 ]; then
	echo "Usage: $0 <uniref_version>  (e.g., 2025_01)"
	exit 1
fi

# === Configuration ===
UNIVERSION="$1"
BASEURL="https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-${UNIVERSION}/uniref/"
ARCHIVE="uniref${UNIVERSION}.tar.gz"
TARGET_TAR="uniref50.tar"
TARGET_XML_GZ="uniref50.xml.gz"
RENAMED_XML_GZ="uniref50_${UNIVERSION}.xml.gz"

# === Download ===
echo "📥 Downloading $ARCHIVE ..."
curl -O "$BASEURL/$ARCHIVE"

# === Extract uniref50.tar only ===
echo "📦 Extracting $TARGET_TAR from $ARCHIVE ..."
tar -xzvf "$ARCHIVE" "$TARGET_TAR"

# === Extract uniref50.xml.gz from uniref50.tar ===
echo "📂 Extracting $TARGET_XML_GZ from $TARGET_TAR ..."
tar -xvf "$TARGET_TAR" "$TARGET_XML_GZ"

# === Rename the output to include the version ===
mv "$TARGET_XML_GZ" "$RENAMED_XML_GZ"

# === Cleanup large intermediate files ===
echo "🧹 Cleaning up large files ..."
rm -f "$ARCHIVE"
rm -f "$TARGET_TAR"

# === Done ===
echo "✅ Finished. Output file: $RENAMED_XML_GZ"
