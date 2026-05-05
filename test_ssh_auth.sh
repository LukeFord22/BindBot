#!/bin/bash
#
# SSH Authentication Test Script
# Runs detailed diagnostics on SSH publickey authentication failure
#

set -euo pipefail

echo "=== SSH Authentication Deep Dive ==="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "  ${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

check_warn() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

echo "[1] Filesystem Checks:"
if [ -d /root/.ssh ]; then
    PERM=$(stat -c %a /root/.ssh)
    if [ "$PERM" = "700" ]; then
        check_pass "/root/.ssh permissions: $PERM (correct)"
    else
        check_fail "/root/.ssh permissions: $PERM (should be 700)"
        echo "    Fix: chmod 700 /root/.ssh"
    fi
else
    check_fail "/root/.ssh directory does not exist"
    echo "    Fix: mkdir -p /root/.ssh && chmod 700 /root/.ssh"
fi

if [ -f /root/.ssh/authorized_keys ]; then
    PERM=$(stat -c %a /root/.ssh/authorized_keys)
    OWNER=$(stat -c %U:%G /root/.ssh/authorized_keys)
    if [ "$PERM" = "600" ]; then
        check_pass "authorized_keys permissions: $PERM (correct)"
    else
        check_fail "authorized_keys permissions: $PERM (should be 600)"
        echo "    Fix: chmod 600 /root/.ssh/authorized_keys"
    fi

    if [ "$OWNER" = "root:root" ]; then
        check_pass "authorized_keys ownership: $OWNER (correct)"
    else
        check_fail "authorized_keys ownership: $OWNER (should be root:root)"
        echo "    Fix: chown root:root /root/.ssh/authorized_keys"
    fi

    KEYCOUNT=$(wc -l < /root/.ssh/authorized_keys)
    check_pass "authorized_keys contains $KEYCOUNT key(s)"
else
    check_fail "authorized_keys file does not exist"
    echo "    Fix: Create /root/.ssh/authorized_keys with your public key"
fi
echo ""

echo "[2] SSH Daemon Checks:"
if pgrep -x sshd >/dev/null; then
    PIDS=$(pgrep -x sshd | tr '\n' ' ')
    check_pass "sshd is running (PIDs: $PIDS)"
else
    check_fail "sshd is NOT running"
    echo "    Fix: /usr/local/bin/sshd-isolated (or /usr/sbin/sshd)"
fi

if netstat -tlnp 2>/dev/null | grep -q ':22 '; then
    check_pass "Port 22 is listening"
    netstat -tlnp 2>/dev/null | grep ':22 ' | sed 's/^/    /'
else
    check_fail "Port 22 is NOT listening"
fi
echo ""

echo "[3] SSH Configuration Checks:"
if /usr/sbin/sshd -t 2>/dev/null; then
    check_pass "sshd config is valid"
else
    check_fail "sshd config has errors:"
    /usr/sbin/sshd -t 2>&1 | sed 's/^/    /'
fi

# Check critical config values
PERMIT_ROOT=$(grep "^PermitRootLogin" /etc/ssh/sshd_config | awk '{print $2}')
PUBKEY_AUTH=$(grep "^PubkeyAuthentication" /etc/ssh/sshd_config | awk '{print $2}')
PASS_AUTH=$(grep "^PasswordAuthentication" /etc/ssh/sshd_config | awk '{print $2}')
AUTH_KEYS_FILE=$(grep "^AuthorizedKeysFile" /etc/ssh/sshd_config | awk '{print $2}')

if [ "$PERMIT_ROOT" = "yes" ]; then
    check_pass "PermitRootLogin: yes"
else
    check_fail "PermitRootLogin: $PERMIT_ROOT (should be yes)"
fi

if [ "$PUBKEY_AUTH" = "yes" ]; then
    check_pass "PubkeyAuthentication: yes"
else
    check_fail "PubkeyAuthentication: $PUBKEY_AUTH (should be yes)"
fi

if [ "$PASS_AUTH" = "no" ]; then
    check_pass "PasswordAuthentication: no"
else
    check_warn "PasswordAuthentication: $PASS_AUTH (should be no for key-only auth)"
fi

if [ -n "$AUTH_KEYS_FILE" ]; then
    check_pass "AuthorizedKeysFile: $AUTH_KEYS_FILE"
else
    check_warn "AuthorizedKeysFile not explicitly set (using default)"
fi
echo ""

echo "[4] Library Conflict Checks:"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    check_warn "LD_LIBRARY_PATH is set: $LD_LIBRARY_PATH"
    echo "    This can cause OpenSSL version conflicts"
    echo "    Use /usr/local/bin/sshd-isolated wrapper to fix"
else
    check_pass "LD_LIBRARY_PATH is not set"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
    check_warn "CONDA_PREFIX is set: $CONDA_PREFIX"
    echo "    May cause library conflicts with sshd"
else
    check_pass "CONDA_PREFIX is not set"
fi
echo ""

echo "[5] Key Format Validation:"
if command -v ssh-keygen >/dev/null 2>&1 && [ -f /root/.ssh/authorized_keys ]; then
    echo "  Checking key format with ssh-keygen..."
    if ssh-keygen -l -f /root/.ssh/authorized_keys >/dev/null 2>&1; then
        check_pass "All keys are valid SSH format"
        ssh-keygen -l -f /root/.ssh/authorized_keys | sed 's/^/    /'
    else
        check_fail "One or more keys are INVALID"
        ssh-keygen -l -f /root/.ssh/authorized_keys 2>&1 | sed 's/^/    /'
    fi
else
    check_warn "Cannot validate key format (ssh-keygen or file missing)"
fi
echo ""

echo "[6] PAM Configuration:"
if grep -q "^UsePAM yes" /etc/ssh/sshd_config; then
    check_warn "UsePAM is enabled"
    echo "    PAM might be blocking authentication"
    echo "    Check /etc/pam.d/sshd for restrictive rules"
    if [ -f /etc/pam.d/sshd ]; then
        echo "    Active PAM rules:"
        grep -v "^#" /etc/pam.d/sshd | grep -v "^$" | sed 's/^/      /'
    fi
else
    check_pass "UsePAM is disabled or not set"
fi
echo ""

echo "=== Authentication Test ==="
echo ""

if command -v ssh >/dev/null 2>&1; then
    echo "[Test 1] Attempting local SSH connection..."
    echo "  Command: ssh -o StrictHostKeyChecking=no -o PasswordAuthentication=no root@localhost 'echo SUCCESS'"
    echo ""

    if timeout 5 ssh -o StrictHostKeyChecking=no -o PasswordAuthentication=no root@localhost 'echo SUCCESS' 2>&1; then
        check_pass "SSH authentication SUCCESSFUL!"
    else
        check_fail "SSH authentication FAILED"
        echo ""
        echo "  Trying with verbose output..."
        timeout 5 ssh -vvv -o StrictHostKeyChecking=no -o PasswordAuthentication=no root@localhost 'echo SUCCESS' 2>&1 | grep -E "(debug1|Offering|Authentications|Permission)" | tail -n 20 | sed 's/^/    /'
    fi
else
    check_warn "ssh client not available for testing"
fi

echo ""
echo "=== Recommendations ==="
echo ""

ISSUES_FOUND=0

if [ ! -f /root/.ssh/authorized_keys ]; then
    echo "❌ CRITICAL: No authorized_keys file"
    echo "   Fix: echo 'your-public-key' > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"
    ISSUES_FOUND=1
fi

if ! pgrep -x sshd >/dev/null; then
    echo "❌ CRITICAL: sshd is not running"
    echo "   Fix: /usr/local/bin/sshd-isolated (or /usr/sbin/sshd)"
    ISSUES_FOUND=1
fi

if [ -n "${LD_LIBRARY_PATH:-}" ] && ! [ -x /usr/local/bin/sshd-isolated ]; then
    echo "⚠️  WARNING: Library path conflicts detected"
    echo "   Fix: Use the sshd-isolated wrapper (created by entrypoint)"
    ISSUES_FOUND=1
fi

if [ "$ISSUES_FOUND" -eq 0 ]; then
    echo "✅ All basic checks passed!"
    echo ""
    echo "If SSH still fails, try running sshd in debug mode:"
    echo "  pkill sshd"
    echo "  /usr/sbin/sshd -d -p 2222"
    echo ""
    echo "Then connect with:"
    echo "  ssh -vvv root@localhost -p 2222"
    echo ""
    echo "This will show detailed authentication logs."
fi

echo ""
