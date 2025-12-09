import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# model_dir = 'Llama-3p1-8B-Instruct'

# ============================
# 0) 配置：在这里填你的目录
# ============================
# 每个目录下面有若干个 CSV，每个 CSV 里有 total_ttft、avg_score 两列
DIRS = [
    f"intermediate_results/{model_dir}/results"
]

# ============================
# 1) 收集所有 CSV 并读取
# ============================
series_list = []  # 每个元素： (df, label)

for d in DIRS:
    csv_paths = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not csv_paths:
        print(f"[WARN] No CSV found under {d}")
        continue

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if not {"total_ttft", "avg_score"}.issubset(df.columns):
            print(f"[SKIP] {csv_path}: missing total_ttft / avg_score")
            continue

        # label: 目录名 + 文件名（不含扩展名）
        dir_name = os.path.basename(os.path.normpath(d))
        file_stem = os.path.splitext(os.path.basename(csv_path))[0]
        label = f"{dir_name}/{file_stem}"

        print(f"[LOAD] {csv_path}  -> label = {label}")
        series_list.append((df, label))

if not series_list:
    print("No valid CSVs found. Please check DIRS.")
else:
    # ============================
    # 2) 画图（风格参考你给的 cell）
    # ============================
    fig, ax = plt.subplots()

    for df, label in series_list:
        # 丢掉缺失值并按 total_ttft 排序
        df_plot = (
            df[["total_ttft", "avg_score"]]
            .dropna(subset=["total_ttft", "avg_score"])
            .sort_values("total_ttft")
        )

        if df_plot.empty:
            print(f"[SKIP] {label}: empty after dropping NaN")
            continue

        ax.plot(
            df_plot["total_ttft"],
            df_plot["avg_score"],
            marker="o",
            label=label,
        )

    ax.set_xlabel("Total TTFT (s)")
    ax.set_ylabel("Average score")
    ax.set_title("Quality vs. TTFT (from multiple CSVs)")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=3,
        )

    plt.subplots_adjust(top=0.75)

    ax.grid(True)
    ax.set_xlim(left=0)
    # ax.set_xlim(right=20000)
    ax.set_ylim(bottom=0)

    # ==== 先保存，再 show ====
    out_dir = f"intermediate_results/{model_dir}"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "quality_vs_ttft.png")
    fig.savefig(out_path, dpi=300)
    print(f"[SAVE] Figure saved to {out_path}")

    plt.show()
