# SSH Troubleshooting Guide for RunPod

## Current Issue: Permission Denied (publickey)

Despite correct setup (sshd running, keys present, correct permissions), SSH authentication fails with "Permission denied (publickey)".

## Quick Diagnosis

Run the debug script inside the container:
```bash
bash /tmp/diagnose_ssh.sh
```

Or run the comprehensive debug script:
```bash
bash debug_ssh.sh
```

## Common Issues & Fixes

### Issue 1: OpenSSL Version Mismatch

**Symptom:**
```
OpenSSL version mismatch. Built against 30000020, you have 30600020
```

**Cause:** Conda's LD_LIBRARY_PATH causes sshd to load conda's OpenSSL instead of system OpenSSL.

**Fix:** The updated entrypoint now creates `/usr/local/bin/sshd-isolated` wrapper that isolates sshd from conda libraries.

**Manual fix if needed:**
```bash
cat > /usr/local/bin/sshd-isolated << 'EOF'
#!/bin/bash
unset LD_LIBRARY_PATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
exec /usr/sbin/sshd "$@"
EOF
chmod +x /usr/local/bin/sshd-isolated

# Restart sshd
pkill sshd
/usr/local/bin/sshd-isolated
```

### Issue 2: Duplicate Subsystem sftp

**Symptom:**
```
/etc/ssh/sshd_config line 133: Subsystem 'sftp' already defined.
```

**Fix:** The Dockerfile now removes duplicates before adding the Subsystem line.

**Manual fix if needed:**
```bash
sed -i '/Subsystem.*sftp/d' /etc/ssh/sshd_config
echo "Subsystem sftp /usr/lib/openssh/sftp-server" >> /etc/ssh/sshd_config
sshd -t  # Test config
```

### Issue 3: Keys Not Persisting (RunPod Restarts)

**Symptom:** authorized_keys disappears after pod restart.

**Cause:** RunPod doesn't persist `/root/.ssh` by default. Only `/workspace` is persistent.

**Fix:** Store keys in persistent volume:
```bash
mkdir -p /workspace/.ssh
chmod 700 /workspace/.ssh

# Copy or create your authorized_keys in /workspace/.ssh/
cp /root/.ssh/authorized_keys /workspace/.ssh/authorized_keys
chmod 600 /workspace/.ssh/authorized_keys

# Symlink to /root/.ssh
rm -rf /root/.ssh
ln -s /workspace/.ssh /root/.ssh
```

### Issue 4: PAM Blocking Authentication

**Symptom:** publickey auth fails even with correct setup.

**Diagnosis:**
```bash
cat /etc/pam.d/sshd
```

**Potential fix:** Disable PAM in sshd_config:
```bash
sed -i 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
pkill sshd
/usr/local/bin/sshd-isolated
```

### Issue 5: File Permissions

**Required permissions:**
- `/root/.ssh` → 700 (drwx------)
- `/root/.ssh/authorized_keys` → 600 (-rw-------)

**Fix:**
```bash
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
```

## Advanced Debugging

### Run sshd in Debug Mode

This shows exactly what sshd is checking and why it's rejecting authentication:

```bash
# Stop running sshd
pkill sshd

# Run on alternate port in debug mode
/usr/sbin/sshd -d -p 2222

# In another terminal/SSH session:
ssh -vvv root@localhost -p 2222
```

**What to look for in debug output:**
- "debug1: trying public key file /root/.ssh/authorized_keys"
- "Authentication refused: bad ownership or modes"
- "debug1: restore_uid: 0/0"
- Any permission errors

### Check sshd Can Read authorized_keys

```bash
# Use strace to see file access
strace -e open,openat /usr/sbin/sshd -D -d 2>&1 | grep authorized_keys

# Or test manually as root
su root -c "cat /root/.ssh/authorized_keys"
```

### Verify Key Format

```bash
# Check key is valid SSH format
ssh-keygen -l -f /root/.ssh/authorized_keys

# Should output something like:
# 2048 SHA256:xxxxx user@host (RSA)
# 256 SHA256:xxxxx user@host (ED25519)
```

### Check SSH Logs

```bash
# Ubuntu/Debian
tail -f /var/log/auth.log | grep sshd

# Or use journalctl if available
journalctl -u sshd -f

# Or check for any SSH-related logs
find /var/log -name "*ssh*" -o -name "*auth*"
```

## RunPod-Specific Considerations

### SSH Key Injection

RunPod typically injects SSH keys via:
1. **Environment variable:** `PUBLIC_KEY` environment variable
2. **Automatic injection:** RunPod's infrastructure automatically adds keys

The entrypoint checks for `PUBLIC_KEY` and adds it to authorized_keys if found.

### Port Exposure

RunPod uses a TCP proxy for SSH. Ensure:
1. Port 22 is exposed in Dockerfile: `EXPOSE 22`
2. RunPod template has TCP port 22 configured
3. Use the connection command provided by RunPod (not direct IP:22)

### Connection Method

**Correct RunPod SSH connection:**
```bash
ssh root@<pod-id>.runpod.io -p <assigned-port> -i ~/.ssh/your_key
```

**Not:**
```bash
ssh root@<pod-ip>:22  # This won't work through RunPod's proxy
```

## Testing Checklist

- [ ] sshd is running (`pgrep -x sshd`)
- [ ] Port 22 is listening (`netstat -tlnp | grep :22`)
- [ ] authorized_keys exists and has content
- [ ] Permissions are correct (700 for dir, 600 for file)
- [ ] sshd config is valid (`sshd -t`)
- [ ] No OpenSSL version mismatch errors
- [ ] Using RunPod's provided connection command
- [ ] Key matches the one in RunPod account settings

## Manual Key Addition

If RunPod's automatic key injection isn't working:

```bash
# Add your public key manually
echo "ssh-rsa AAAA... your-key-here user@host" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# Restart sshd
pkill sshd
/usr/local/bin/sshd-isolated
```

## Last Resort: Rebuild with Diagnostics

If nothing works, rebuild the image with enhanced logging:

```dockerfile
# In Dockerfile, modify entrypoint to enable sshd debug mode
CMD ["/usr/local/bin/sshd-isolated", "-D", "-d"]
```

This will show all authentication attempts in the container logs.

## Getting Help

If you're still stuck:

1. Run `debug_ssh.sh` and save the output
2. Run sshd in debug mode and capture output
3. Check RunPod's documentation for SSH access
4. Verify your SSH key is correctly added in RunPod account settings
5. Check RunPod community forums for similar issues
