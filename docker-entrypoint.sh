#!/usr/bin/env bash
#
# BindBot Base Environment Entrypoint
# Clones/updates GitHub repo at runtime, then executes command
#

set -euo pipefail

echo "=== BindBot Base Environment Entrypoint ==="
echo "[INFO] Current directory: $(pwd)"
echo "[INFO] Directory contents: $(ls -la /app 2>/dev/null | wc -l) items"

# Configuration from environment variables
REPO_URL="${GITHUB_REPO:-https://github.com/lukeford22/BindBot.git}"
BRANCH="${GITHUB_BRANCH:-main}"
APP_DIR="/app"

# Increase file descriptor limits (BindCraft can open many files)
if command -v prlimit >/dev/null 2>&1; then
  prlimit --pid $$ --nofile=65536:65536 2>/dev/null || true
else
  ulimit -n 65536 2>/dev/null || true
fi

# Function to clone repository
clone_repo() {
    echo "[INFO] Cloning repository: $REPO_URL (branch: $BRANCH)"
    cd /
    rm -rf "${APP_DIR}.tmp"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "${APP_DIR}.tmp" || {
        echo "[ERROR] Failed to clone repository"
        exit 1
    }

    # Move contents (handles .git and all hidden files)
    mv "${APP_DIR}.tmp"/* "${APP_DIR}.tmp"/.[!.]* "$APP_DIR/" 2>/dev/null || true
    rmdir "${APP_DIR}.tmp"

    cd "$APP_DIR"
    echo "[SUCCESS] Repository cloned successfully"
    git log -1 --oneline
}

# Function to update repository
update_repo() {
    echo "[INFO] Updating existing repository"
    cd "$APP_DIR"

    # Stash any local changes
    if ! git diff --quiet 2>/dev/null; then
        echo "[WARN] Local changes detected, stashing..."
        git stash 2>/dev/null || true
    fi

    # Pull latest
    git fetch origin "$BRANCH" || {
        echo "[ERROR] Failed to fetch updates"
        return 1
    }
    git reset --hard "origin/$BRANCH" || {
        echo "[ERROR] Failed to update to latest"
        return 1
    }

    echo "[SUCCESS] Repository updated successfully"
    git log -1 --oneline
}

# Main logic: Check if repo exists and handle accordingly
if [ ! -d "$APP_DIR/.git" ]; then
    echo "[STEP] No git repository found in $APP_DIR"

    # Check if directory is empty or only has minimal files
    if [ -z "$(ls -A $APP_DIR 2>/dev/null)" ] || [ $(ls -A $APP_DIR | wc -l) -lt 3 ]; then
        echo "[STEP] Directory is empty, cloning fresh repository..."
        clone_repo
    else
        echo "[WARN] Directory not empty but no .git found"
        echo "[WARN] Contents: $(ls -la $APP_DIR)"
        echo "[STEP] Backing up and cloning fresh..."
        mkdir -p /tmp/app_backup
        mv "$APP_DIR"/* "$APP_DIR"/.[!.]* /tmp/app_backup/ 2>/dev/null || true
        clone_repo
    fi
else
    echo "[STEP] Git repository found in $APP_DIR"

    # Optionally update if AUTOUPDATE is set
    if [ "${AUTOUPDATE:-false}" = "true" ]; then
        echo "[STEP] AUTOUPDATE=true, pulling latest changes..."
        update_repo || {
            echo "[WARN] Update failed, continuing with existing code"
        }
    else
        echo "[INFO] Using existing repository (set AUTOUPDATE=true to auto-pull)"
        cd "$APP_DIR"
        git log -1 --oneline 2>/dev/null || echo "[WARN] Could not read git log"
    fi
fi

# Create symlinks from /data to /app so code finds binaries/weights
echo "[STEP] Setting up symlinks for baked-in resources..."
cd "$APP_DIR"

# Link params if not already present
if [ ! -e "$APP_DIR/params" ]; then
    ln -sf /data/params "$APP_DIR/params"
    echo "[INFO] Linked /data/params -> /app/params"
fi

# Link functions if not already present (prefer repo version, fallback to baked)
if [ ! -d "$APP_DIR/functions" ] || [ ! "$(ls -A $APP_DIR/functions 2>/dev/null)" ]; then
    # No functions in repo, use baked version
    rm -rf "$APP_DIR/functions"
    ln -sf /data/functions "$APP_DIR/functions"
    echo "[INFO] Linked /data/functions -> /app/functions (using baked binaries)"
else
    echo "[INFO] Using functions/ from repository"
    # Still ensure binaries from /data are available if repo is missing them
    for binary in dssp sc FASPR DAlphaBall.gcc; do
        if [ ! -f "$APP_DIR/functions/$binary" ] && [ -f "/data/functions/$binary" ]; then
            ln -sf "/data/functions/$binary" "$APP_DIR/functions/$binary"
            echo "[INFO] Linked /data/functions/$binary -> /app/functions/$binary"
        fi
    done
fi

# Verify critical files exist
echo "[STEP] Verifying BindCraft installation..."
if [ ! -f "$APP_DIR/bindcraft.py" ]; then
    echo "[ERROR] bindcraft.py not found in repository!"
    echo "[ERROR] Repository may be incomplete or corrupt"
    exit 1
fi

if [ ! -f "/data/params/params_model_5_ptm.npz" ]; then
    echo "[ERROR] AlphaFold weights not found at /data/params/"
    echo "[ERROR] Base image may be corrupt"
    exit 1
fi

echo "[SUCCESS] BindCraft environment ready"
echo "[INFO] Repository: $REPO_URL"
echo "[INFO] Current commit: $(git -C $APP_DIR log -1 --oneline 2>/dev/null || echo 'unknown')"
echo "[INFO] AlphaFold weights: /data/params/ ($(du -sh /data/params 2>/dev/null | cut -f1))"
echo "[INFO] Binaries: /data/functions/"
echo ""

# Execute the provided command
cd "$APP_DIR"
echo "[INFO] Executing command: $@"
echo "=========================================="

# If no command provided OR command is bash/sh, keep container alive
if [ $# -eq 0 ] || [ "$1" = "/bin/bash" ] || [ "$1" = "bash" ] || [ "$1" = "/bin/sh" ] || [ "$1" = "sh" ]; then
    echo "[INFO] Starting container in persistent mode for SSH access..."

    # Ensure SSH directories and permissions are correct
    mkdir -p /var/run/sshd /root/.ssh
    chmod 700 /root/.ssh

    # Regenerate host keys if needed
    ssh-keygen -A >/dev/null 2>&1 || true

    # Fix authorized_keys permissions if it exists
    if [ -f /root/.ssh/authorized_keys ]; then
        chmod 600 /root/.ssh/authorized_keys
        echo "[INFO] Found authorized_keys with $(wc -l < /root/.ssh/authorized_keys) keys"
    else
        echo "[WARN] No authorized_keys found at /root/.ssh/authorized_keys"
        echo "[WARN] RunPod should inject keys automatically. If SSH fails, check RunPod SSH key settings."
    fi

    # Check if SSH public key exists in environment (RunPod injects this)
    if [ -n "${PUBLIC_KEY:-}" ]; then
        echo "[INFO] Found PUBLIC_KEY environment variable, adding to authorized_keys..."
        echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        echo "[SUCCESS] Added key from PUBLIC_KEY environment variable"
    fi

    # Start SSH daemon if not already running
    if ! pgrep -x sshd >/dev/null; then
        echo "[INFO] Starting sshd on port 22..."
        # Unset LD_LIBRARY_PATH to prevent library conflicts
        env -u LD_LIBRARY_PATH /usr/sbin/sshd -D &
        SSHD_PID=$!
        echo "[SUCCESS] sshd started (PID: $SSHD_PID)"

        # Give sshd a moment to start
        sleep 2

        # Verify it's running
        if pgrep -x sshd >/dev/null; then
            echo "[SUCCESS] sshd is running and accepting connections"
        else
            echo "[ERROR] sshd failed to start!"
        fi
    else
        echo "[INFO] sshd already running"
    fi

    echo ""
    echo "========================================"
    echo "Container ready for SSH access"
    echo "Connect using RunPod's SSH instructions"
    echo "========================================"
    echo ""

    # Keep container alive
    exec tail -f /dev/null
else
    exec "$@"
fi
