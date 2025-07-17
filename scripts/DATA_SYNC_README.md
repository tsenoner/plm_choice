# Data Synchronization

Sync project data between local machine and TUM Nextcloud using `scripts/sync_data.sh`.

## Quick Start

```bash
# Upload local data to Nextcloud
./scripts/sync_data.sh upload

# Download data from Nextcloud
./scripts/sync_data.sh download

# Check sync status
./scripts/sync_data.sh status
```

## Commands

| Command     | Description                          |
| ----------- | ------------------------------------ |
| `upload`    | Upload local data to Nextcloud       |
| `download`  | Download data from Nextcloud         |
| `status`    | Show local and remote data sizes     |
| `sync`      | Bidirectional sync (use carefully)   |
| `configure` | Setup rclone (auto-runs when needed) |

## What Gets Synced

✅ **Included:**

- `data/processed/` (~35GB)
- `data/interm/` (~21GB)

❌ **Excluded:**

- `data/raw/` (too large, 127GB)
- Temporary files (`*.tmp`, `*.temp`)

## Workflow

1. **Setup:** Clone repository
2. **Get data:** `./scripts/sync_data.sh download`
3. **Work:** Make changes to code/data
4. **Share:** `./scripts/sync_data.sh upload`
5. **Verify:** `./scripts/sync_data.sh status`

## Troubleshooting

**Connection issues:**

```bash
rclone ls tum_nextcloud:  # Test connection
```

**Missing rclone:** The script will prompt for installation if needed.

**Sync conflicts:** Use `status` command to check both local and remote state before syncing.

---

📍 Script location: `scripts/sync_data.sh`
🔗 Repository: [https://github.com/tsenoner/unknown_unknowns](https://github.com/tsenoner/unknown_unknowns)
