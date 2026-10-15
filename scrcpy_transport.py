"""Video and multi-touch client for the pinned official scrcpy 5.0.1 server."""
from __future__ import annotations

import secrets
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parent
SCRCPY = ROOT / "tools" / "scrcpy-win64-v5.0.1"
ADB = SCRCPY / "adb.exe"
VERSION = "5.0.1"


def adb(*args, serial=None, timeout=20):
    cmd = [str(ADB)]
    if serial:
        cmd += ["-s", serial]
    result = subprocess.run(cmd + list(args), capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def connected_device():
    rows = adb("devices").decode().splitlines()[1:]
    devices = [row.split()[0] for row in rows if len(row.split()) >= 2 and row.split()[1] == "device"]
    if len(devices) != 1:
        raise RuntimeError(f"Connect exactly one authorized Android device; found {len(devices)}.")
    return devices[0]


def recv_exact(sock, length):
    data = bytearray()
    while len(data) < length:
        block = sock.recv(length - len(data))
        if not block:
            raise ConnectionError("scrcpy stream disconnected")
        data.extend(block)
    return bytes(data)


def touch_packet(action, pointer, x, y, width, height):
    if action not in (0, 1, 2):
        raise ValueError("Invalid touch action")
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("Touch lies outside the frame")
    return struct.pack(
        ">BBQiiHHHII", 2, action, pointer, x, y, width, height,
        0 if action == 1 else 65535, 0, 0,
    )


class ScrcpySession:
    def __init__(self, serial=None, max_size=1600, max_fps=30, tap_hold_ms=55):
        if not 5<=tap_hold_ms<=150:raise ValueError('tap_hold_ms must be between 5 and 150')
        self.tap_hold_seconds=tap_hold_ms/1000
        self.serial = serial or connected_device()
        self.max_size = max_size
        self.max_fps = max_fps
        self.video = self.control = self.process = None
        self.port = None
        self.frame = None
        self.frame_time = 0.0
        self.frame_lock = threading.Lock()
        self.frame_condition = threading.Condition(self.frame_lock)
        self.frame_sequence = 0
        self.frame_metrics = {}
        self.touch_lock = threading.RLock()
        self.tap_condition = threading.Condition(self.touch_lock)
        self.tap_deadlines = {}
        self.tap_worker = None
        self.pointers = {}
        self.pointer_started = {}
        self.stopped = threading.Event()
        self.error = None
        self.width = self.height = 0
        self.thread = None
        (ROOT / "logs").mkdir(exist_ok=True)
        self.log = None

    def start(self):
        scid = f"{secrets.randbits(31):08x}"
        jar = f"/data/local/tmp/laya-scrcpy-{scid}.jar"
        adb("push", str(SCRCPY / "scrcpy-server"), jar, serial=self.serial)
        self.port = int(adb("forward", "tcp:0", f"localabstract:scrcpy_{scid}", serial=self.serial))
        options = [
            f"scid={scid}", "tunnel_forward=true", "audio=false", "control=true",
            "clipboard_autosync=false", "send_device_meta=false", "video_codec=h264",
            f"max_size={self.max_size}", f"max_fps={self.max_fps}", "video_bit_rate=4000000",
        ]
        self.log = (ROOT / "logs" / "scrcpy-server.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(ADB), "-s", self.serial, "shell", f"CLASSPATH={jar}", "app_process", "/",
             "com.genymobile.scrcpy.Server", VERSION, *options],
            stdout=self.log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.perf_counter() + 15
            while time.perf_counter() < deadline:
                try:
                    self.video = socket.create_connection(("127.0.0.1", self.port), timeout=1)
                    recv_exact(self.video, 1)  # ADB forward handshake
                    break
                except (OSError, ConnectionError):
                    if self.video:
                        self.video.close()
                        self.video = None
                    if self.process.poll() is not None:
                        raise RuntimeError("scrcpy server failed; see logs/scrcpy-server.log")
                    time.sleep(0.1)
            if not self.video:
                raise TimeoutError("scrcpy handshake timed out")
            self.control = socket.create_connection(("127.0.0.1", self.port), timeout=3)
            codec_id = recv_exact(self.video, 4)
            if codec_id != b"h264":
                raise RuntimeError(f"Unexpected video codec {codec_id!r}")
            self.video.settimeout(10)
            self.thread = threading.Thread(target=self._decode, name="scrcpy-video", daemon=True)
            self.thread.start()
            while time.perf_counter() < deadline:
                if self.frame is not None:
                    return self
                if self.error:
                    raise RuntimeError(self.error)
                time.sleep(0.05)
            raise TimeoutError("No video frame received from scrcpy")
        except Exception:
            self.close()
            raise

    def _decode(self):
        decoder = None
        config = b""
        try:
            while not self.stopped.is_set():
                header = recv_exact(self.video, 12)
                first, width, height = struct.unpack(">III", header)
                if first & 0x80000000:
                    self.release_all()
                    self.width, self.height = width, height
                    decoder = av.CodecContext.create("h264", "r")
                    decoder.thread_count = 1
                    config = b""
                    continue
                flags, size = struct.unpack(">QI", header)
                if not 0 < size <= 16 * 1024 * 1024:
                    raise RuntimeError("Invalid scrcpy packet length")
                payload = recv_exact(self.video, size)
                received_at = time.perf_counter()
                if flags & (1 << 62):
                    config += payload
                    continue
                if decoder is None:
                    raise RuntimeError("Video packet arrived before session metadata")
                packet = av.Packet(config + payload)
                config = b""
                for decoded in decoder.decode(packet):
                    frame = decoded.to_ndarray(format="bgr24")
                    with self.frame_condition:
                        self.frame = frame
                        self.frame_time = time.perf_counter()
                        self.frame_sequence += 1
                        self.frame_metrics = {"received_at":received_at,"decoded_at":self.frame_time,
                                              "source_pts_us":flags & ((1 << 62)-1),"sequence":self.frame_sequence}
                        self.frame_condition.notify_all()
        except Exception as exc:
            if not self.stopped.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
                self.release_all()

    def latest(self):
        with self.frame_lock:
            # Published arrays are immutable: observers/encoders only read them.
            return self.frame, self.frame_time

    def wait_frame(self, after_sequence, timeout=.01):
        with self.frame_condition:
            self.frame_condition.wait_for(lambda:self.frame_sequence>after_sequence or self.stopped.is_set(),timeout)
            return self.frame_sequence

    def touch(self, action, pointer, point):
        with self.touch_lock:
            if not self.control or not self.width or not self.height:
                return
            if action == 1:
                self.tap_deadlines.pop(pointer,None)
                previous = self.pointers.pop(pointer, None)
                if previous is None:
                    return
                self.pointer_started.pop(pointer,None)
                x, y, width, height = previous
            else:
                x = min(self.width - 1, max(0, round(point[0] * self.width)))
                y = min(self.height - 1, max(0, round(point[1] * self.height)))
                width, height = self.width, self.height
                if action == 2 and pointer not in self.pointers:
                    action = 0
                if action == 0:
                    self.pointer_started[pointer] = time.perf_counter()
                self.pointers[pointer] = (x, y, width, height)
            self.control.sendall(touch_packet(action, pointer, x, y, width, height))
            self.last_control_sent_at = time.perf_counter()

    def tap(self, point, pointer=1):
        """Send DOWN now; schedule UP without holding the controller thread.

        Stop/release_all cancels pending pulses and releases immediately. Reusing a
        pointer replaces its pending deadline; an old timer cannot release a new tap.
        """
        with self.tap_condition:
            if pointer in self.pointers:self.touch(1,pointer,(0,0))
            self.touch(0,pointer,point)
            if pointer not in self.pointers:return
            self.tap_deadlines[pointer]=time.perf_counter()+self.tap_hold_seconds
            if self.tap_worker is None:
                self.tap_worker=threading.Thread(target=self._release_taps,name='scrcpy-tap-release',daemon=True)
                self.tap_worker.start()
            self.tap_condition.notify_all()

    def _release_taps(self):
        with self.tap_condition:
            while not self.stopped.is_set():
                if not self.tap_deadlines:
                    self.tap_condition.wait(.1);continue
                pointer,deadline=min(self.tap_deadlines.items(),key=lambda item:item[1])
                remaining=deadline-time.perf_counter()
                if remaining>0:
                    self.tap_condition.wait(remaining);continue
                try:self.touch(1,pointer,(0,0))
                except OSError as exc:
                    self.error=f'Control: {exc}';self.tap_deadlines.clear();return

    def release_all(self):
        with self.touch_lock:
            self.tap_deadlines.clear()
            for pointer in list(self.pointers):
                try:
                    self.touch(1, pointer, (0, 0))
                except OSError:
                    self.pointers.pop(pointer, None)

    def close(self):
        self.stopped.set()
        with self.tap_condition:self.tap_condition.notify_all()
        with self.frame_condition:self.frame_condition.notify_all()
        self.release_all()
        for sock in (self.control, self.video):
            if sock:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
        if self.tap_worker and self.tap_worker is not threading.current_thread():
            self.tap_worker.join(timeout=1)
        if self.process:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=2)
        if self.port:
            adb("forward", "--remove", f"tcp:{self.port}", serial=self.serial)
        if self.log:
            self.log.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
