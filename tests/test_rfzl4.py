from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import struct
import numpy as np

from rfzl4 import (
    CHUNK_HEADER_SIZE,
    FILE_HEADER_SIZE,
    FLAG_TIMESTAMPS_PRESENT,
    RawLZ4FrameReader,
    RawLZ4FrameWriter,
    RFZL4CorruptionError,
    RFZL4FormatError,
    read_file_header,
)
from rfzl4.container import (
    CHUNK_MAGIC,
    COLOR_FORMAT_BGR,
    DTYPE_CODE_UINT8,
    FORMAT_VERSION,
    MAGIC,
    ChunkHeader,
    FileHeader,
)


def _make_frame(i: int) -> np.ndarray:
    frame = np.empty((256, 256, 3), dtype=np.uint8)
    frame[..., 0] = i % 256
    frame[..., 1] = (i * 7) % 256
    frame[..., 2] = (i * 13) % 256
    return frame


def test_header_binary_sizes_are_fixed() -> None:
    assert FILE_HEADER_SIZE == 128
    assert CHUNK_HEADER_SIZE == 64
    assert struct.calcsize("<4sHH" + ("I" * 10) + ("Q" * 5) + "40s") == FILE_HEADER_SIZE
    assert struct.calcsize("<4sHHIIIIIQQQI8s") == CHUNK_HEADER_SIZE


def test_roundtrip_and_partial_last_chunk(tmp_path: Path) -> None:
    path = tmp_path / "session.rfzl4"
    writer = RawLZ4FrameWriter(path, chunk_frames=64)

    src_frames: list[np.ndarray] = []
    src_ts: list[int] = []
    for i in range(70):
        frame = _make_frame(i)
        ts = 1_700_000_000_000_000_000 + i
        src_frames.append(frame.copy())
        src_ts.append(ts)
        writer.write_frame(frame, ts)

    final_path = writer.close()
    assert final_path.endswith(".rfzl4")
    assert Path(final_path).exists()
    assert not Path(f"{final_path}.tmp").exists()

    header = read_file_header(path)
    assert header.magic == MAGIC
    assert header.version == FORMAT_VERSION
    assert header.header_size == FILE_HEADER_SIZE
    assert header.total_frames == 70
    assert header.total_chunks == 2
    assert header.frame_size == 256 * 256 * 3

    dst: list[tuple[int, np.ndarray]] = []
    with RawLZ4FrameReader(path) as reader:
        for item in reader:
            dst.append(item)

    assert len(dst) == 70
    for i, (ts, frame) in enumerate(dst):
        assert ts == src_ts[i]
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8
        assert np.array_equal(frame, src_frames[i])


def test_non_finalized_tmp_file_is_rejected(tmp_path: Path) -> None:
    tmp_file = tmp_path / "incomplete.rfzl4.tmp"
    header = FileHeader(
        magic=MAGIC,
        version=FORMAT_VERSION,
        header_size=FILE_HEADER_SIZE,
        flags=FLAG_TIMESTAMPS_PRESENT,
        width=256,
        height=256,
        channels=3,
        dtype_code=DTYPE_CODE_UINT8,
        color_format=COLOR_FORMAT_BGR,
        nominal_fps=25,
        frame_size=256 * 256 * 3,
        chunk_frames=64,
        max_frames=0,
        session_start_ns=1,
        session_end_ns=0,
        total_frames=0,
        total_chunks=0,
        index_offset=0,
    )
    tmp_file.write_bytes(header.pack())

    with pytest.raises(RFZL4FormatError):
        RawLZ4FrameReader(tmp_file)


def test_corrupted_chunk_skip_vs_strict(tmp_path: Path) -> None:
    path = tmp_path / "corrupted.rfzl4"
    writer = RawLZ4FrameWriter(path, chunk_frames=64)
    for i in range(130):
        writer.write_frame(_make_frame(i), 2_000_000_000 + i)
    writer.close()

    with open(path, "r+b") as fh:
        fh.seek(FILE_HEADER_SIZE)
        first_chunk = ChunkHeader.unpack(fh.read(CHUNK_HEADER_SIZE))
        assert first_chunk.chunk_magic == CHUNK_MAGIC
        corrupted = replace(first_chunk, timestamps_size=first_chunk.timestamps_size + 8)
        fh.seek(FILE_HEADER_SIZE)
        fh.write(corrupted.pack())

    with RawLZ4FrameReader(path, skip_corrupted=True) as reader:
        frames_skip = list(reader)
    assert len(frames_skip) == 66

    with pytest.raises(RFZL4CorruptionError):
        with RawLZ4FrameReader(path, skip_corrupted=False) as reader:
            list(reader)
