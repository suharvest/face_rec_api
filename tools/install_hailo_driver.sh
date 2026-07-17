#!/usr/bin/env bash
# install_hailo_driver.sh — provision the Hailo-8 PCIe *kernel* driver +
# firmware on a fresh Debian/Raspberry Pi OS (bookworm/trixie) host.
#
# This is ALL the host needs for the containerized face_rec_api:hailo
# deployment: the userland library (libhailort.so.4.21.0) and the Python
# binding are baked into the image (see Dockerfile.hailo). Do NOT install
# the host `hailort` userland package for the container path.
#
# Detection-first behavior (never blindly reinstalls):
#   * driver already present at 4.21.x  -> report "already in place", exit 0,
#     NO changes made.
#   * driver present but wrong version  -> loud ABI-mismatch warning, exit 1,
#     driver left untouched. Only `--force-reinstall` proceeds to
#     uninstall + reinstall 4.21.0.
#   * no driver                         -> full install: kernel headers check,
#     DKMS build of v4.21.0 (github.com/hailo-ai/hailort-drivers), hailo8
#     firmware into /lib/firmware/hailo/, udev rule, modprobe. DKMS pins the
#     exact source tag and auto-rebuilds on kernel updates.
#
# Options:
#   --dry-run           only report what was detected / what would be done
#   --force-reinstall   allow replacing a mismatched existing driver
#
# Version pinning: the DKMS path builds exactly tag v4.21.0, nothing to hold.
# If the driver instead came from the RPi apt archive (`hailort-pcie-driver`),
# pin it:  sudo apt-mark hold hailort-pcie-driver
#
# Usage:  sudo ./install_hailo_driver.sh [--dry-run] [--force-reinstall]
set -euo pipefail

HAILORT_VER="4.21.0"
WANT_MM="4.21"                       # major.minor that must match the image
DRIVER_REPO="https://github.com/hailo-ai/hailort-drivers"

DRY_RUN=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --dry-run)         DRY_RUN=1 ;;
        --force-reinstall) FORCE=1 ;;
        *) echo "Unknown option: $arg" >&2
           echo "Usage: sudo $0 [--dry-run] [--force-reinstall]" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Phase 1 — DETECT existing driver (three sources, first hit wins)
# ---------------------------------------------------------------------------
detected_ver=""
detected_src=""
if modinfo hailo_pci >/dev/null 2>&1; then
    detected_ver="$(modinfo -F version hailo_pci 2>/dev/null || true)"
    detected_src="kernel module (modinfo hailo_pci)"
fi
if [ -z "$detected_ver" ] && command -v dpkg-query >/dev/null 2>&1; then
    pkg_ver="$(dpkg-query -W -f='${Version}' hailort-pcie-driver 2>/dev/null || true)"
    if [ -n "$pkg_ver" ]; then
        detected_ver="$pkg_ver"
        detected_src="apt package hailort-pcie-driver"
    fi
fi
if [ -z "$detected_ver" ] && command -v dkms >/dev/null 2>&1; then
    dkms_line="$(dkms status 2>/dev/null | grep -i hailo | head -1 || true)"
    if [ -n "$dkms_line" ]; then
        detected_ver="$(echo "$dkms_line" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
        detected_src="dkms ($dkms_line)"
    fi
fi

echo "== Hailo driver detection =="
if [ -n "$detected_ver" ]; then
    echo "   found:   v${detected_ver}  via ${detected_src}"
else
    echo "   found:   none"
fi
echo "   wanted:  ${HAILORT_VER} (major.minor ${WANT_MM}.x must match the"
echo "            libhailort baked into face_rec_api:hailo)"

# --- Branch 1: already in place at 4.21.x → skip, no changes ---------------
if [ -n "$detected_ver" ] && [ "${detected_ver%.*}" = "$WANT_MM" ]; then
    echo "OK: driver already in place (v${detected_ver}) — skipping install, no changes made."
    exit 0
fi

# --- Branch 2: present but wrong version → warn, refuse unless --force -----
if [ -n "$detected_ver" ]; then
    echo "" >&2
    echo "!! WARNING: installed driver v${detected_ver} does NOT match the" >&2
    echo "!! userland HailoRT ${HAILORT_VER} baked into the container image." >&2
    echo "!! The 4.x ioctl ABI is not cross-version compatible — the service" >&2
    echo "!! will fail at startup with an ABI/version error." >&2
    if [ "$FORCE" -ne 1 ]; then
        echo "!! Leaving the existing driver UNTOUCHED." >&2
        echo "!! Re-run with --force-reinstall to uninstall it and install ${HAILORT_VER}." >&2
        exit 1
    fi
    echo "!! --force-reinstall given: existing v${detected_ver} will be removed." >&2
fi

# --- Branch 3 (and forced 2): install -------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] would do:"
    [ "$FORCE" -eq 1 ] && [ -n "$detected_ver" ] \
        && echo "[dry-run]  - remove existing driver v${detected_ver} (${detected_src})"
    echo "[dry-run]  - ensure kernel headers for $(uname -r)"
    echo "[dry-run]  - apt install git dkms build-essential curl"
    echo "[dry-run]  - DKMS build+install ${DRIVER_REPO} tag v${HAILORT_VER}"
    echo "[dry-run]  - install hailo8 firmware -> /lib/firmware/hailo/hailo8_fw.bin"
    echo "[dry-run]  - install udev rule + modprobe hailo_pci, verify /dev/hailo0"
    exit 0
fi

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run with sudo/root" >&2; exit 1; }
WORK_DIR="$(mktemp -d /tmp/hailort-drivers.XXXXXX)"

# forced removal of a mismatched driver
if [ "$FORCE" -eq 1 ] && [ -n "$detected_ver" ]; then
    modprobe -r hailo_pci 2>/dev/null || true
    dpkg -r hailort-pcie-driver 2>/dev/null || true
    if command -v dkms >/dev/null 2>&1; then
        dkms status 2>/dev/null | grep -i hailo | cut -d, -f1-2 | tr -d ' ' | \
            while IFS=/ read -r mod ver; do dkms remove "${mod}/${ver}" --all || true; done
    fi
fi

# kernel headers (required for the DKMS build)
if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
    echo "Kernel headers for $(uname -r) missing — installing..."
    apt-get update
    # RPi OS ships them as linux-headers-rpi-*; plain Debian as
    # linux-headers-$(uname -r). Try the generic name first.
    apt-get install -y "linux-headers-$(uname -r)" 2>/dev/null \
        || apt-get install -y linux-headers-rpi-v8 linux-headers-rpi-2712 2>/dev/null \
        || { echo "ERROR: could not install kernel headers" >&2; exit 1; }
fi

apt-get update
apt-get install -y --no-install-recommends git dkms build-essential curl

# build + install driver via DKMS
git clone --depth 1 -b "v${HAILORT_VER}" "${DRIVER_REPO}" "${WORK_DIR}"
cd "${WORK_DIR}/linux/pcie"
make install_dkms   # registers hailo_pci/${HAILORT_VER} with DKMS + installs

# firmware
cd "${WORK_DIR}"
./download_firmware.sh          # fetches hailo8_fw.${HAILORT_VER}.bin
mkdir -p /lib/firmware/hailo
mv -f hailo8_fw*.bin /lib/firmware/hailo/hailo8_fw.bin

# udev rule + load
cp "${WORK_DIR}/linux/pcie/51-hailo-udev.rules" /etc/udev/rules.d/
udevadm control --reload-rules && udevadm trigger
modprobe hailo_pci

# verify
sleep 1
if [ -e /dev/hailo0 ]; then
    echo "OK: /dev/hailo0 present, driver $(modinfo -F version hailo_pci)"
else
    echo "Driver installed but /dev/hailo0 not present — is the Hailo-8" \
         "module seated? Check: lspci | grep -i hailo ; dmesg | grep -i hailo" >&2
    exit 1
fi
rm -rf "${WORK_DIR}"
echo "Done. DKMS status: $(dkms status | grep -i hailo || true)"
