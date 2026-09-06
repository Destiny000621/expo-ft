"""Decode selected frames from a recorded video — in a process with NOTHING else in it.

This exists because of a hang, and the hang is worth stating precisely: decoding a
recording with PyAV is fast (~2200 fps) in a fresh process, and deadlocks — futex
wait, 146 threads, no I/O, forever — in a process that has already imported the
learner's stack. That stack pulls in lerobot (and through it a second, differently
built libav via torchcodec) alongside PyAV's own; two libav copies in one process
is a known way to wedge ffmpeg's frame-threading.

So the seeder shells out to this module, which imports only numpy, PIL and av. It
writes an npz keyed by frame index, which the caller loads and deletes.

    python -m expo_ft.env.rollout_decode_worker VIDEO INDICES.npy OUT.npz [--resize 224]
"""

import argparse

import numpy as np

# Decode threads per video. NOT "AUTO": ffmpeg's automatic frame-threading opens one
# thread per core, and the learner box has 224 of them.
DECODE_THREADS = 4


def decode_frames(video: str, wanted: np.ndarray, resize: int = 0) -> dict[str, np.ndarray]:
    import av  # noqa: PLC0415

    want = set(int(i) for i in np.asarray(wanted).reshape(-1))
    out: dict[str, np.ndarray] = {}
    with av.open(video) as container:
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"
        stream.codec_context.thread_count = DECODE_THREADS
        for idx, frame in enumerate(container.decode(stream)):
            if idx in want:
                img = frame.to_ndarray(format="rgb24")
                if resize:
                    # Pad-resize here so a 720p side frame never crosses the process
                    # boundary: it is exactly what the live client sends anyway.
                    from openpi_client import image_tools  # noqa: PLC0415

                    img = image_tools.resize_with_pad(img, resize, resize)
                out[str(idx)] = np.ascontiguousarray(img, dtype=np.uint8)
                if len(out) == len(want):
                    break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("indices")
    p.add_argument("out")
    p.add_argument("--resize", type=int, default=0, help="0 = keep the raw frame")
    args = p.parse_args()
    frames = decode_frames(args.video, np.load(args.indices), args.resize)
    np.savez(args.out, **frames)


if __name__ == "__main__":
    main()
