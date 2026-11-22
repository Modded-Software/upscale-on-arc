"""The Intel GPUs that PyTorch sees, with their PCI addresses."""

from __future__ import annotations

import struct
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from arc_upscale.errors import UserError


class Gpu(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int  # PyTorch's own numbering: this GPU is torch.device("xpu", index)
    name: str
    memory: int
    pci: str | None


def pci_address(uuid: bytes, sysfs: Path = Path("/sys/bus/pci/devices")) -> str | None:
    """PCI address encoded in an Intel Level Zero device UUID, or None.

    Layout, from intel/compute-runtime, shared/source/device/device.cpp
    (struct DeviceUUID in Device::generateUuidFromPciBusInfo):
    vendor u16, device u16, revision u16, domain u16, bus u8, dev u8, func u8.
    The driver can use a chipset-unique UUID instead. So the address is only
    returned if sysfs has a device there with the same vendor and device ID.
    """
    vendor, device, _revision, domain = struct.unpack_from("<4H", uuid)
    address = f"{domain:04x}:{uuid[8]:02x}:{uuid[9]:02x}.{uuid[10]:x}"
    vendor_file, device_file = sysfs / address / "vendor", sysfs / address / "device"
    if not (vendor_file.is_file() and device_file.is_file()):
        return None
    found = (int(vendor_file.read_text(), 16), int(device_file.read_text(), 16))
    return address if found == (vendor, device) else None


def list_gpus() -> list[Gpu]:
    if not torch.xpu.is_available():
        raise UserError(
            "PyTorch finds no Intel GPU. Check the Level Zero driver "
            "(Ubuntu: libze1, libze-intel-gpu1) and your access to /dev/dri/renderD*."
        )
    gpus = []
    for index in range(torch.xpu.device_count()):
        props = torch.xpu.get_device_properties(index)
        pci = pci_address(bytes(props.uuid.bytes))
        gpus.append(Gpu(index=index, name=props.name, memory=props.total_memory, pci=pci))
    return gpus


def pick_gpu(index: int) -> Gpu:
    gpus = list_gpus()
    if not 0 <= index < len(gpus):
        raise UserError(f"GPU {index} does not exist; found {len(gpus)}, numbered from 0 (see: arc-upscale gpus)")
    return gpus[index]
