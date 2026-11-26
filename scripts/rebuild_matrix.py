"""正确重建: head.part[0:60768] (真实 gzip 前缀) + old[60768:] (已下载的 gzip 主体)"""
import os
import zlib

d = r"f:\ai_paper\experiment2_fm_time\data\visium_luad"
old = os.path.join(d, "TD1_IAC", "GSM5702473_TD1_matrix.mtx.gz")
headp = os.path.join(d, "head.part")
out = os.path.join(d, "matrix_v2.mtx.gz")
SKIP = 60768
EXPECT = 54496618

if os.path.exists(out):
    os.remove(out)

with open(headp, "rb") as fh:
    prefix = fh.read(SKIP)
assert len(prefix) == SKIP and prefix[:2] == b"\x1f\x8b", "前缀不是 gzip 头"

with open(old, "rb") as fo, open(out, "wb") as fw:
    fw.write(prefix)
    fo.seek(SKIP)
    while True:
        buf = fo.read(1 << 20)
        if not buf:
            break
        fw.write(buf)

size = os.path.getsize(out)
print("重建后大小:", size, "(期望 %d)" % EXPECT)

n = 0
try:
    dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with open(out, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            n += len(dec.decompress(b))
    print("gzip 校验: OK, 解压输出", n, "字节")
except Exception as e:
    print("gzip 校验: FAIL", e, "| 已解压", n, "字节")
