import subprocess, time, shlex

VIDEO_PATH = "4.mp4"
HOST = "127.0.0.1"   # если смотреть на этой же машине
PORT = 5000

OUT_URL = f"udp://{HOST}:{PORT}?pkt_size=1316"

CMD = [
    "ffmpeg",
    "-hide_banner", "-loglevel", "warning",
    "-re",
    "-stream_loop", "-1",
    "-i", VIDEO_PATH,

    "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
    "-an",
    "-f", "mpegts",
    OUT_URL,
]

while True:
    print("Starting:", " ".join(shlex.quote(x) for x in CMD))
    p = subprocess.Popen(CMD)
    rc = p.wait()
    print("ffmpeg stopped:", rc, "restart in 2s")
    time.sleep(2)
