start = 0
end = 9226714
step = 10000

with open("part_paths.txt", "w", encoding="utf-8") as f:
    for s in range(start, end, step):
        e = min(s + step, end)
        f.write(f"/tos/mnt/processed_parts/part_{s:08d}_{e:08d}.lance\n")
