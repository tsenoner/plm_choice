#!/bin/bash

# Data Synchronization Script for unknown_unknowns project
# Works on both macOS and Linux
# Usage: ./sync_data.sh [upload|download|configure|status]

set -e # Exit on any error

# Configuration
REMOTE_NAME="tum_nextcloud"
NEXTCLOUD_URL="https://nextcloud.in.tum.de/public.php/webdav/"
SHARE_TOKEN="NJoA6BwzXt6KKBK"
LOCAL_DATA_DIR="data"
REMOTE_DATA_DIR="data"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if rclone is installed
check_rclone() {
    if ! command -v rclone &>/dev/null; then
        error "rclone is not installed!"
        echo ""
        echo "Installation instructions:"
        echo ""
        echo "macOS:"
        echo "  brew install rclone"
        echo ""
        echo "Linux (Ubuntu/Debian):"
        echo "  sudo apt update && sudo apt install rclone"
        echo ""
        echo "Linux (CentOS/RHEL):"
        echo "  sudo yum install rclone"
        echo ""
        echo "Or download from: https://rclone.org/downloads/"
        exit 1
    fi
}

# Configure rclone remote
configure_rclone() {
    log "Configuring rclone remote: $REMOTE_NAME"

    # Check if remote already exists
    if rclone listremotes | grep -q "^${REMOTE_NAME}:$"; then
        warning "Remote '$REMOTE_NAME' already exists."
        read -p "Do you want to reconfigure it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Skipping configuration."
            return 0
        fi
        rclone config delete "$REMOTE_NAME"
    fi

    log "Creating new rclone configuration..."

    # Create config file content
    cat >/tmp/rclone_config <<EOF
[$REMOTE_NAME]
type = webdav
url = $NEXTCLOUD_URL
vendor = other
user = $SHARE_TOKEN
EOF

    # Import the configuration
    rclone config file
    CONFIG_FILE=$(rclone config file | tail -n1)
    cat /tmp/rclone_config >>"$CONFIG_FILE"
    rm /tmp/rclone_config

    success "rclone configured successfully!"

    # Test the connection
    log "Testing connection..."
    if rclone ls "$REMOTE_NAME:" >/dev/null 2>&1; then
        success "Connection test successful!"
    else
        error "Connection test failed. Please check your configuration."
        exit 1
    fi
}

# Upload data to Nextcloud
upload_data() {
    log "Starting data upload to Nextcloud..."

    # Check if local data directory exists
    if [ ! -d "$LOCAL_DATA_DIR" ]; then
        error "Local data directory '$LOCAL_DATA_DIR' not found!"
        exit 1
    fi

    # Get size info before upload
    local total_size=$(du -sh "$LOCAL_DATA_DIR" | cut -f1)
    log "Total data to upload: $total_size"

    # Upload processed data (excluding raw)
    if [ -d "$LOCAL_DATA_DIR/processed" ]; then
        log "Uploading processed data..."
        rclone copy "$LOCAL_DATA_DIR/processed" "$REMOTE_NAME:$REMOTE_DATA_DIR/processed" \
            --progress \
            --transfers 4 \
            --checkers 8 \
            --exclude "*.tmp" \
            --exclude "*.temp"
    else
        warning "No processed data directory found"
    fi

    # Upload intermediate data
    if [ -d "$LOCAL_DATA_DIR/interm" ]; then
        log "Uploading intermediate data..."
        rclone copy "$LOCAL_DATA_DIR/interm" "$REMOTE_NAME:$REMOTE_DATA_DIR/interm" \
            --progress \
            --transfers 4 \
            --checkers 8 \
            --exclude "*.tmp" \
            --exclude "*.temp"
    else
        warning "No intermediate data directory found"
    fi

    success "Data upload completed!"
}

# Download data from Nextcloud
download_data() {
    log "Starting data download from Nextcloud..."

    # Create local data directory if it doesn't exist
    mkdir -p "$LOCAL_DATA_DIR"

    # Check what's available on remote
    log "Checking available data on remote..."
    if ! rclone ls "$REMOTE_NAME:$REMOTE_DATA_DIR" >/dev/null 2>&1; then
        error "No data found on remote or connection failed!"
        exit 1
    fi

    # Download processed data
    if rclone ls "$REMOTE_NAME:$REMOTE_DATA_DIR/processed" >/dev/null 2>&1; then
        log "Downloading processed data..."
        rclone copy "$REMOTE_NAME:$REMOTE_DATA_DIR/processed" "$LOCAL_DATA_DIR/processed" \
            --progress \
            --transfers 4 \
            --checkers 8
    else
        warning "No processed data found on remote"
    fi

    # Download intermediate data
    if rclone ls "$REMOTE_NAME:$REMOTE_DATA_DIR/interm" >/dev/null 2>&1; then
        log "Downloading intermediate data..."
        rclone copy "$REMOTE_NAME:$REMOTE_DATA_DIR/interm" "$LOCAL_DATA_DIR/interm" \
            --progress \
            --transfers 4 \
            --checkers 8
    else
        warning "No intermediate data found on remote"
    fi

    success "Data download completed!"
}

# Show sync status
show_status() {
    log "Checking synchronization status..."

    echo ""
    echo "=== LOCAL DATA ==="
    if [ -d "$LOCAL_DATA_DIR" ]; then
        # Show only top-level directories with human-readable sizes
        find "$LOCAL_DATA_DIR" -maxdepth 1 -type d ! -name "$(basename "$LOCAL_DATA_DIR")" -exec du -sh {} \; 2>/dev/null | sort || echo "No local data found"
    else
        echo "No local data directory"
    fi

    echo ""
    echo "=== REMOTE DATA ==="
    if rclone ls "$REMOTE_NAME:" >/dev/null 2>&1; then
        # Show only top-level directories with their sizes
        for dir in processed interm raw; do
            if rclone ls "$REMOTE_NAME:$REMOTE_DATA_DIR/$dir" >/dev/null 2>&1; then
                # Get size in human-readable format
                rclone size "$REMOTE_NAME:$REMOTE_DATA_DIR/$dir" --human-readable 2>/dev/null | grep "Total size:" | awk -v dir="$dir" '{gsub(/\(.*\)/, "", $0); printf "%-10s data/%s\n", $3" "$4, dir}'
            fi
        done
    else
        echo "Cannot connect to remote"
    fi
}

# Sync data (bidirectional)
sync_data() {
    log "Starting bidirectional sync..."

    # This will sync changes in both directions
    if [ -d "$LOCAL_DATA_DIR/processed" ]; then
        log "Syncing processed data..."
        rclone sync "$LOCAL_DATA_DIR/processed" "$REMOTE_NAME:$REMOTE_DATA_DIR/processed" \
            --progress \
            --transfers 4 \
            --checkers 8
    fi

    if [ -d "$LOCAL_DATA_DIR/interm" ]; then
        log "Syncing intermediate data..."
        rclone sync "$LOCAL_DATA_DIR/interm" "$REMOTE_NAME:$REMOTE_DATA_DIR/interm" \
            --progress \
            --transfers 4 \
            --checkers 8
    fi

    success "Bidirectional sync completed!"
}

# Main function
main() {
    check_rclone

    case "${1:-}" in
    "configure")
        configure_rclone
        ;;
    "upload")
        configure_rclone
        upload_data
        ;;
    "download")
        configure_rclone
        download_data
        ;;
    "sync")
        configure_rclone
        sync_data
        ;;
    "status")
        show_status
        ;;
    "help" | "-h" | "--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  configure  Set up rclone configuration"
        echo "  upload     Upload local data to Nextcloud"
        echo "  download   Download data from Nextcloud to local"
        echo "  sync       Bidirectional sync (be careful!)"
        echo "  status     Show local and remote data status"
        echo "  help       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 configure    # Set up rclone first time"
        echo "  $0 upload       # Upload your data to server"
        echo "  $0 download     # Download data on server"
        echo "  $0 status       # Check what's where"
        ;;
    *)
        error "Unknown command: ${1:-}"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
    esac
}

# Run main function with all arguments
main "$@"
