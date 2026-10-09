"""
Resource Monitor
Reads the current system state (RAM, CPU) so the router can make hard-constraint
and scoring decisions based on what's actually available right now, not just
theoretical model requirements.

Usage (standalone test):
    python resource_monitor.py

Usage (from other code):
    from resource_monitor import get_system_state
    state = get_system_state()
"""

import psutil


def get_system_state() -> dict:
    """
    Returns a snapshot of current system resources.
    Called by the router before scoring candidate models.
    """
    vm = psutil.virtual_memory()

    state = {
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "ram_available_gb": round(vm.available / (1024 ** 3), 2),
        "ram_used_percent": vm.percent,
        "cpu_percent": psutil.cpu_percent(interval=0.5),  # 0.5s sample for a stable reading
    }

    # Battery info - not all systems report this (e.g. desktops), so handle gracefully
    battery = psutil.sensors_battery()
    if battery is not None:
        state["battery_percent"] = battery.percent
        state["battery_charging"] = battery.power_plugged
    else:
        state["battery_percent"] = None
        state["battery_charging"] = None

    # GPU note, not a live reading: Ollama's own startup log on this machine shows
    # "dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1" - meaning the AMD
    # Radeon integrated graphics is NOT used for inference at all on this hardware.
    # Rather than fabricate a GPU utilization number nothing is actually measuring,
    # this is recorded as a documented fact: all inference here is CPU-only.
    state["gpu_used_for_inference"] = False
    state["gpu_note"] = "Integrated GPU not used by Ollama on this hardware - CPU-only inference"

    return state


def print_state(state: dict):
    print("=== CURRENT SYSTEM STATE ===")
    print(f"RAM available : {state['ram_available_gb']} GB / {state['ram_total_gb']} GB total")
    print(f"RAM used      : {state['ram_used_percent']}%")
    print(f"CPU load      : {state['cpu_percent']}%")
    if state["battery_percent"] is not None:
        charging = "charging" if state["battery_charging"] else "on battery"
        print(f"Battery       : {state['battery_percent']}% ({charging})")
    else:
        print("Battery       : not available (desktop or unsupported)")
    print(f"GPU           : {state['gpu_note']}")


if __name__ == "__main__":
    state = get_system_state()
    print_state(state)