#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3C 分析：1.5B 与 2.9B 的逐层敏感度排序是否一致（按相对深度配对）。

两层模型层数不同（1.5B=24、2.9B=32），按归一化深度 d = i/(L-1) 配对：
对 1.5B 的第 j 层（d=j/23），取 2.9B 深度最接近的层 i = round(d*31)。
然后看两件事：
  1) 配对后的 Spearman（逐层 KL 的相关性）
  2) 各自最敏感/最安全的层落在什么深度

用法: python p3c_compare_depths.py
"""

import json


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    kl15 = {int(k): v["kl_mean"] for k, v in
            json.load(open("sensitivity_15b.json"))["targets"].items() if k != "head"}
    kl29 = {int(k): v["kl_mean"] for k, v in
            json.load(open("sensitivity_p1.json"))["targets"].items() if k != "head"}
    n15, n29 = len(kl15), len(kl29)
    print("层数：1.5B=%d，2.9B=%d" % (n15, n29))

    pairs = []
    for j in sorted(kl15):
        d = j / (n15 - 1)
        i = round(d * (n29 - 1))
        pairs.append((j, i, d, kl15[j], kl29[i]))

    print("\n%-8s %-8s %-8s %-10s %-10s" % ("1.5B层", "深度", "2.9B层", "1.5B KL", "2.9B KL"))
    for j, i, d, a, b in pairs:
        print("%-8d %-8.2f %-8d %-10.4f %-10.4f" % (j, d, i, a, b))

    rho = spearman([p[3] for p in pairs], [p[4] for p in pairs])
    print("\n按相对深度配对后的 Spearman = %.3f（%d 对）" % (rho, len(pairs)))

    for name, kl, n in (("1.5B", kl15, n15), ("2.9B", kl29, n29)):
        order = sorted(kl, key=lambda k: kl[k])
        safe = [(k, round(k / (n - 1), 2)) for k in order[:6]]
        sens = [(k, round(k / (n - 1), 2)) for k in order[-6:]]
        print("\n%s 最安全 6 层（层, 深度）：%s" % (name, safe))
        print("%s 最敏感 6 层（层, 深度）：%s" % (name, sens))
        print("%s 最敏感层位于深度 %.2f，最敏感 6 层的深度跨度 %.2f~%.2f"
              % (name, order[-1] / (n - 1), order[-6] / (n - 1), order[-1] / (n - 1)))


if __name__ == "__main__":
    main()
