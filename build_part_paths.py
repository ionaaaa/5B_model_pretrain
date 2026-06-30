start = 150000
end = 1150000
step = 10000

with open("part_paths.txt", "w", encoding="utf-8") as f:
    for s in range(start, end, step):
        e = s + step
        f.write(f"s3://jianyu/AAAI2026/processed_parts/part_{s:08d}_{e:08d}.lance/\n")
