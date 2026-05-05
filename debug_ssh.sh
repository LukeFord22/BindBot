#!/bin/bash
#
# SSH Debugging Script for RunPod
# Run this inside the container to diagnose SSH issues
#

echo "=== SSH Debugging Report ==="
echo ""

echo "[1] SSH Daemon Status:"
if pgrep -x sshd >/dev/null; then
    echo "  ✓ sshd is running"
    echo "  PIDs: $(pgrep -x sshd | tr '\n' ' ')"
else
    echo "  ✗ sshd is NOT running"
    echo "  Try: /usr/sbin/sshd"
fi
echo ""

echo "[2] SSH Port Status:"
if netstat -tlnp 2>/dev/null | grep -q ':22 '; then
    echo "  ✓ Port 22 is listening"
    netstat -tlnp 2>/dev/null | grep ':22 '
else
    echo "  ✗ Port 22 is NOT listening"
fi
echo ""

echo "[3] Authorized Keys:"
if [ -f /root/.ssh/authorized_keys ]; then
    echo "  ✓ authorized_keys exists"
    echo "  Location: /root/.ssh/authorized_keys"
    echo "  Permissions: $(ls -l /root/.ssh/authorized_keys)"
    echo "  Number of keys: $(wc -l < /root/.ssh/authorized_keys)"
    echo "  First key (truncated): $(head -n 1 /root/.ssh/authorized_keys | cut -c 1-60)..."
else
    echo "  ✗ authorized_keys NOT found"
    echo "  Expected location: /root/.ssh/authorized_keys"
fi
echo ""

echo "[4] SSH Directory Permissions:"
echo "  /root/.ssh: $(ls -ld /root/.ssh 2>/dev/null || echo 'MISSING')"
if [ -f /root/.ssh/authorized_keys ]; then
    echo "  /root/.ssh/authorized_keys: $(ls -l /root/.ssh/authorized_keys)"
fi
echo ""

echo "[5] Environment Variables (RunPod injected):"
if [ -n "$PUBLIC_KEY" ]; then
    echo "  ✓ PUBLIC_KEY is set ($(echo $PUBLIC_KEY | cut -c 1-50)...)"
else
    echo "  ✗ PUBLIC_KEY not found in environment"
fi

if [ -n "$RUNPOD_POD_ID" ]; then
    echo "  ✓ RUNPOD_POD_ID: $RUNPOD_POD_ID"
else
    echo "  ⚠ RUNPOD_POD_ID not set (not running on RunPod?)"
fi
echo ""

echo "[6] SSH Config:"
echo "  PermitRootLogin: $(grep -i '^PermitRootLogin' /etc/ssh/sshd_config || echo 'not set')"
echo "  PubkeyAuthentication: $(grep -i '^PubkeyAuthentication' /etc/ssh/sshd_config || echo 'not set')"
echo "  PasswordAuthentication: $(grep -i '^PasswordAuthentication' /etc/ssh/sshd_config || echo 'not set')"
echo "  AuthorizedKeysFile: $(grep -i '^AuthorizedKeysFile' /etc/ssh/sshd_config || echo 'not set')"
echo ""

echo "[7] SSH Config Validation:"
if /usr/sbin/sshd -t 2>&1; then
    echo "  ✓ sshd config is valid"
else
    echo "  ✗ sshd config has errors (see above)"
fi
echo ""

echo "[8] Library Conflicts Check:"
if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "  ⚠ LD_LIBRARY_PATH is set: $LD_LIBRARY_PATH"
    echo "  This may cause OpenSSL version conflicts with sshd"
else
    echo "  ✓ LD_LIBRARY_PATH is not set"
fi
echo ""

echo "[9] SSH Logs (last 20 lines):"
if [ -f /var/log/auth.log ]; then
    tail -n 20 /var/log/auth.log | grep -i ssh
elif [ -f /var/log/secure ]; then
    tail -n 20 /var/log/secure | grep -i ssh
else
    echo "  No SSH logs found (try: journalctl -u sshd)"
fi
echo ""

echo "[10] Key Format Validation:"
if command -v ssh-keygen >/dev/null && [ -f /root/.ssh/authorized_keys ]; then
    echo "  Validating keys with ssh-keygen..."
    ssh-keygen -l -f /root/.ssh/authorized_keys 2>&1 | head -n 5
else
    echo "  Cannot validate (ssh-keygen or authorized_keys missing)"
fi
echo ""

echo "[11] Test SSH Connection (from localhost):"
if command -v ssh >/dev/null; then
    echo "  Testing: ssh -o StrictHostKeyChecking=no root@localhost -p 22 'echo OK'"
    timeout 3 ssh -o StrictHostKeyChecking=no root@localhost -p 22 'echo OK' 2>&1 || echo "  ✗ Connection failed"
else
    echo "  ssh client not available for testing"
fi
echo ""

echo "=== Recommended Actions ==="
if [ ! -f /root/.ssh/authorized_keys ]; then
    echo "1. Check RunPod SSH key settings in your account"
    echo "2. Try setting PUBLIC_KEY environment variable in RunPod template"
    echo "3. Manual fix: echo 'your-ssh-public-key' > /root/.ssh/authorized_keys"
fi

if ! pgrep -x sshd >/dev/null; then
    echo "1. Start sshd manually: /usr/sbin/sshd"
fi

echo ""
echo "=== Quick Fixes ==="
echo ""
echo "# Manually add your SSH key:"
echo "echo 'ssh-rsa AAAA...' > /root/.ssh/authorized_keys"
echo "chmod 600 /root/.ssh/authorized_keys"
echo ""
echo "# Restart sshd:"
echo "pkill sshd"
echo "/usr/sbin/sshd"
echo ""
