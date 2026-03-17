from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np
import zstandard as zstd

MAGIC = b"RFZS"
CHUNK_MAGIC = b"CHK1"
FORMAT_VERSION = 1

FILE_HEADER_SIZE = 128
CHUNK_HEADER_SIZE = 64

DTYPE_CODE_UINT8 = 1
COLOR_FORMAT_BGR = 1

FLAG_TIMESTAMPS_PRESENT = 1 << 0
FLAG_FILE_FINALIZED = 1 << 1
FLAG_CHUNK_CHECKSUMS_PRESENT = 1 << 2
FLAG_CHUNK_INDEX_PRESENT = 1 << 3

_FILE_HEADER_STRUCT = struct.Struct("<4sHH" + ("I" * 10) + ("Q" * 5) + "40s")
_CHUNK_HEADER_STRUCT = struct.Struct("<4sHHIIIIIQQQI8s")

if _FILE_HEADER_STRUCT.size != FILE_HEADER_SIZE:
    raise RuntimeError(f"File header size mismatch: {_FILE_HEADER_STRUCT.size} != {FILE_HEADER_SIZE}")
if _CHUNK_HEADER_STRUCT.size != CHUNK_HEADER_SIZE:
    raise RuntimeError(f"Chunk header size mismatch: {_CHUNK_HEADER_STRUCT.size} != {CHUNK_HEADER_SIZE}")


class RFZL4FormatError(Exception):
    pass


class RFZL4CorruptionError(Exception):
    pass


@dataclass(frozen=True)
class FileHeader:
    magic: bytes
    version: int
    header_size: int
    flags: int
    width: int
    height: int
    channels: int
    dtype_code: int
    color_format: int
    nominal_fps: int
    frame_size: int
    chunk_frames: int
    max_frames: int
    session_start_ns: int
    session_end_ns: int
    total_frames: int
    total_chunks: int
    index_offset: int
    reserved: bytes = b"\x00" * 40

    def pack(self) -> bytes:
        return _FILE_HEADER_STRUCT.pack(
            self.magic,
            self.version,
            self.header_size,
            self.flags,
            self.width,
            self.height,
            self.channels,
            self.dtype_code,
            self.color_format,
            self.nominal_fps,
            self.frame_size,
            self.chunk_frames,
            self.max_frames,
            self.session_start_ns,
            self.session_end_ns,
            self.total_frames,
            self.total_chunks,
            self.index_offset,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "FileHeader":
        if len(data) != FILE_HEADER_SIZE:
            raise RFZL4FormatError(f"Invalid file header length: {len(data)}")
        values = _FILE_HEADER_STRUCT.unpack(data)
        return cls(*values)


@dataclass(frozen=True)
class ChunkHeader:
    chunk_magic: bytes
    chunk_header_size: int
    chunk_flags: int
    chunk_index: int
    frame_count: int
    raw_size: int
    timestamps_size: int
    compressed_size: int
    first_frame_no: int
    first_ts_ns: int
    last_ts_ns: int
    checksum_xxh32: int
    reserved: bytes = b"\x00" * 8

    def pack(self) -> bytes:
        return _CHUNK_HEADER_STRUCT.pack(
            self.chunk_magic,
            self.chunk_header_size,
            self.chunk_flags,
            self.chunk_index,
            self.frame_count,
            self.raw_size,
            self.timestamps_size,
            self.compressed_size,
            self.first_frame_no,
            self.first_ts_ns,
            self.last_ts_ns,
            self.checksum_xxh32,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ChunkHeader":
        if len(data) != CHUNK_HEADER_SIZE:
            raise RFZL4FormatError(f"Invalid chunk header length: {len(data)}")
        values = _CHUNK_HEADER_STRUCT.unpack(data)
        return cls(*values)


def read_file_header(path: str | Path) -> FileHeader:
    with open(path, "rb") as fh:
        data = fh.read(FILE_HEADER_SIZE)
    return FileHeader.unpack(data)


class RawZstdFrameWriter:
    def __init__(
        self,
        path: str | Path,
        width: int = 256,
        height: int = 256,
        channels: int = 3,
        fps: int = 25,
        chunk_frames: int = 64,
        zstd_level: int = 3,
    ):
        if width <= 0 or height <= 0 or channels <= 0:
            raise ValueError("width, height and channels must be positive")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")

        self.final_path = Path(path)
        self.tmp_path = Path(f"{self.final_path}.tmp")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)

        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.fps = int(fps)
        self.chunk_frames = int(chunk_frames)
        self.zstd_level = int(zstd_level)
        self.frame_size = self.width * self.height * self.channels
        self.session_start_ns = time.time_ns()
        self._compressor = zstd.ZstdCompressor(level=self.zstd_level)

        self._flags = FLAG_TIMESTAMPS_PRESENT
        self._total_frames = 0
        self._total_chunks = 0
        self._buffer_timestamps: list[int] = []
        self._buffer_frames: list[bytes] = []
        self._closed = False

        self._fh = open(self.tmp_path, "wb+")
        self._write_file_header(finalized=False, session_end_ns=0)

    def _write_file_header(self, finalized: bool, session_end_ns: int) -> None:
        flags = self._flags
        if finalized:
            flags |= FLAG_FILE_FINALIZED

        header = FileHeader(
            magic=MAGIC,
            version=FORMAT_VERSION,
            header_size=FILE_HEADER_SIZE,
            flags=flags,
            width=self.width,
            height=self.height,
            channels=self.channels,
            dtype_code=DTYPE_CODE_UINT8,
            color_format=COLOR_FORMAT_BGR,
            nominal_fps=self.fps,
            frame_size=self.frame_size,
            chunk_frames=self.chunk_frames,
            max_frames=0,
            session_start_ns=self.session_start_ns,
            session_end_ns=session_end_ns,
            total_frames=self._total_frames,
            total_chunks=self._total_chunks,
            index_offset=0,
        )
        self._fh.seek(0)
        self._fh.write(header.pack())

    def write_frame(self, frame: np.ndarray, timestamp_ns: int) -> None:
        if self._closed:
            raise RuntimeError("Writer is already closed")
        if frame.shape != (self.height, self.width, self.channels):
            raise ValueError(
                f"Invalid frame shape {frame.shape}, expected {(self.height, self.width, self.channels)}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(f"Invalid frame dtype {frame.dtype}, expected uint8")

        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        self._buffer_timestamps.append(int(timestamp_ns))
        self._buffer_frames.append(frame.tobytes(order="C"))

        if len(self._buffer_frames) >= self.chunk_frames:
            self._flush_chunk()

    def _flush_chunk(self) -> None:
        frame_count = len(self._buffer_frames)
        if frame_count == 0:
            return

        timestamps = np.asarray(self._buffer_timestamps, dtype="<u8")
        timestamps_blob = timestamps.tobytes(order="C")
        frames_blob = b"".join(self._buffer_frames)
        payload = timestamps_blob + frames_blob
        compressed = self._compressor.compress(payload)

        chunk_header = ChunkHeader(
            chunk_magic=CHUNK_MAGIC,
            chunk_header_size=CHUNK_HEADER_SIZE,
            chunk_flags=0,
            chunk_index=self._total_chunks,
            frame_count=frame_count,
            raw_size=len(frames_blob),
            timestamps_size=len(timestamps_blob),
            compressed_size=len(compressed),
            first_frame_no=self._total_frames,
            first_ts_ns=int(self._buffer_timestamps[0]),
            last_ts_ns=int(self._buffer_timestamps[-1]),
            checksum_xxh32=0,
        )

        self._fh.seek(0, os.SEEK_END)
        self._fh.write(chunk_header.pack())
        self._fh.write(compressed)

        self._total_frames += frame_count
        self._total_chunks += 1
        self._buffer_timestamps.clear()
        self._buffer_frames.clear()

    def close(self) -> str:
        if self._closed:
            return str(self.final_path)

        self._flush_chunk()
        session_end_ns = time.time_ns()
        self._write_file_header(finalized=True, session_end_ns=session_end_ns)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        os.replace(self.tmp_path, self.final_path)
        self._closed = True
        return str(self.final_path)

    def __enter__(self) -> "RawZstdFrameWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close()


class RawZstdFrameReader:
    def __init__(self, path: str | Path, skip_corrupted: bool = True):
        self.path = Path(path)
        self.skip_corrupted = bool(skip_corrupted)
        self._fh: BinaryIO = open(self.path, "rb")
        header_data = self._fh.read(FILE_HEADER_SIZE)
        self.file_header = FileHeader.unpack(header_data)
        self._validate_header(self.file_header)
        self._decompressor = zstd.ZstdDecompressor()

        self.width = self.file_header.width
        self.height = self.file_header.height
        self.channels = self.file_header.channels
        self.frame_size = self.file_header.frame_size

    def _validate_header(self, header: FileHeader) -> None:
        if header.magic != MAGIC:
            raise RFZL4FormatError(f"Invalid magic: {header.magic!r}")
        if header.version != FORMAT_VERSION:
            raise RFZL4FormatError(f"Unsupported format version: {header.version}")
        if header.header_size != FILE_HEADER_SIZE:
            raise RFZL4FormatError(f"Unsupported file header size: {header.header_size}")
        if not (header.flags & FLAG_FILE_FINALIZED):
            raise RFZL4FormatError("File is not finalized")
        if header.dtype_code != DTYPE_CODE_UINT8:
            raise RFZL4FormatError(f"Unsupported dtype_code: {header.dtype_code}")
        if header.color_format != COLOR_FORMAT_BGR:
            raise RFZL4FormatError(f"Unsupported color_format: {header.color_format}")

        expected_frame_size = header.width * header.height * header.channels
        if header.frame_size != expected_frame_size:
            raise RFZL4FormatError(
                f"Invalid frame_size {header.frame_size}, expected {expected_frame_size}"
            )

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        self._fh.seek(FILE_HEADER_SIZE)
        while True:
            chunk_header_data = self._fh.read(CHUNK_HEADER_SIZE)
            if not chunk_header_data:
                return
            if len(chunk_header_data) != CHUNK_HEADER_SIZE:
                if self.skip_corrupted:
                    return
                raise RFZL4CorruptionError("Truncated chunk header")

            chunk = ChunkHeader.unpack(chunk_header_data)
            if chunk.chunk_magic != CHUNK_MAGIC:
                if self.skip_corrupted:
                    return
                raise RFZL4CorruptionError(f"Invalid chunk magic: {chunk.chunk_magic!r}")
            if chunk.chunk_header_size != CHUNK_HEADER_SIZE:
                if self.skip_corrupted:
                    return
                raise RFZL4CorruptionError(f"Unsupported chunk header size: {chunk.chunk_header_size}")

            compressed = self._fh.read(chunk.compressed_size)
            if len(compressed) != chunk.compressed_size:
                if self.skip_corrupted:
                    return
                raise RFZL4CorruptionError("Truncated compressed chunk payload")

            expected_timestamps_size = chunk.frame_count * 8
            expected_raw_size = chunk.frame_count * self.frame_size
            expected_payload_size = expected_timestamps_size + expected_raw_size

            try:
                payload = self._decompressor.decompress(compressed, max_output_size=expected_payload_size)
            except Exception as exc:  # noqa: BLE001
                if self.skip_corrupted:
                    continue
                raise RFZL4CorruptionError(f"Failed to decompress chunk {chunk.chunk_index}") from exc

            if len(payload) != expected_payload_size:
                if self.skip_corrupted:
                    continue
                raise RFZL4CorruptionError(
                    f"Invalid payload size {len(payload)} for chunk {chunk.chunk_index}, expected {expected_payload_size}"
                )
            if chunk.timestamps_size != expected_timestamps_size or chunk.raw_size != expected_raw_size:
                if self.skip_corrupted:
                    continue
                raise RFZL4CorruptionError(
                    f"Inconsistent chunk sizes in header for chunk {chunk.chunk_index}"
                )

            timestamps = np.frombuffer(payload, dtype="<u8", count=chunk.frame_count, offset=0)
            data_offset = expected_timestamps_size

            for idx in range(chunk.frame_count):
                frame_offset = data_offset + idx * self.frame_size
                frame = np.frombuffer(
                    payload,
                    dtype=np.uint8,
                    count=self.frame_size,
                    offset=frame_offset,
                ).reshape((self.height, self.width, self.channels))
                yield int(timestamps[idx]), frame

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RawZstdFrameReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
