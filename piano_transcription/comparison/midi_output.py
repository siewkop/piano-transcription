from contextlib import contextmanager
from ctypes import POINTER, Structure, WinDLL, byref, c_size_t, create_unicode_buffer, sizeof
from ctypes.wintypes import DWORD, HANDLE, LPWSTR, UINT, WCHAR, WORD
from dataclasses import dataclass
from threading import Event
from time import perf_counter

class MidiOutCaps(Structure):
    _fields_ = [
        ("manufacturer_id", WORD),
        ("product_id", WORD),
        ("driver_version", UINT),
        ("name", WCHAR * 32),
        ("technology", WORD),
        ("voices", WORD),
        ("notes", WORD),
        ("channel_mask", WORD),
        ("support", DWORD),
    ]


winmm = WinDLL("winmm")
for name, argtypes in (
    ("midiOutGetNumDevs", []),
    ("midiOutGetDevCapsW", [c_size_t, POINTER(MidiOutCaps), UINT]),
    ("midiOutGetErrorTextW", [UINT, LPWSTR, UINT]),
    ("midiOutOpen", [POINTER(HANDLE), UINT, c_size_t, c_size_t, DWORD]),
    ("midiOutShortMsg", [HANDLE, DWORD]),
    ("midiOutReset", [HANDLE]),
    ("midiOutClose", [HANDLE]),
):
    function = getattr(winmm, name)
    function.argtypes = argtypes
    function.restype = UINT

MOD_MIDIPORT = 1


@dataclass(frozen=True)
class MidiPort:
    device_id: int
    name: str
    hardware: bool


@dataclass(frozen=True)
class Song:
    midi_bytes: bytes
    events: tuple[tuple[float, bytes], ...]
    duration: float


def check_midi_result(result: int) -> None:
    if result:
        message = create_unicode_buffer(256)
        winmm.midiOutGetErrorTextW(result, message, len(message))
        raise OSError(message.value or f"Błąd Windows MIDI ({result}).")


def output_ports() -> list[MidiPort]:
    ports = []
    for device_id in range(winmm.midiOutGetNumDevs()):
        caps = MidiOutCaps()
        if winmm.midiOutGetDevCapsW(device_id, byref(caps), sizeof(caps)) == 0:
            ports.append(MidiPort(device_id, caps.name, caps.technology == MOD_MIDIPORT))
    return ports


@contextmanager
def open_midi_output(port: MidiPort):
    if port not in output_ports():
        raise OSError("Wyjście MIDI jest niedostępne. Podłącz interfejs i odśwież listę.")
    handle = HANDLE()
    check_midi_result(winmm.midiOutOpen(byref(handle), port.device_id, 0, 0, 0))
    try:
        yield handle
    finally:
        winmm.midiOutClose(handle)


def play_song(song: Song, port: MidiPort, stop: Event) -> bool:
    """Send MIDI messages and return True if the last event was played."""
    with open_midi_output(port) as handle:
        finished = False
        try:
            started = perf_counter()
            for timestamp, message in song.events:
                if stop.wait(max(0.0, started + timestamp - perf_counter())):
                    return False
                # WinMM stores the status byte in the least significant byte.
                packed_message = int.from_bytes(message, byteorder="little")
                check_midi_result(winmm.midiOutShortMsg(handle, packed_message))
            finished = True
        finally:
            if not finished:
                winmm.midiOutReset(handle)
    return finished
