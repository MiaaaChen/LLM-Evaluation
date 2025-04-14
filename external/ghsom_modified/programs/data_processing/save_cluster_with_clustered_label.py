import numpy as np
import pandas as pd
from fractions import Fraction
import argparse
import get_ghsom_dim
import os

# 解析命令列參數
parser = argparse.ArgumentParser(description="manual to this script")
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--tau1", type=float, default=0.1)
parser.add_argument("--tau2", type=float, default=0.01)
parser.add_argument("--index", type=str, default=None)
args = parser.parse_args()

# 初始化參數
prefix = args.name
t1 = args.tau1
t2 = args.tau2
index = args.index
file = f"{prefix}-{t1}-{t2}"

# 獲取 GHSOM 層級資訊
layers, max_layer, number_of_digits = get_ghsom_dim.layers(file)

# 載入原始資料
df_source = pd.read_csv(f"./raw-data/{prefix}.csv", encoding="utf-8")
median = df_source.iloc[:, 1:].median(axis=1)
mean = df_source.iloc[:, 1:].mean(axis=1)
df_source["mean"] = mean
df_source["median"] = median
df_source["clustered_label"] = np.nan
df_source["x_y_label"] = np.nan

for i in range(1, max_layer + 1):
    df_source[f"clusterL{i}"] = np.nan


# 獲取節點標記
def get_cluster_flag(text_file):
    get_cluster_flag = [i for i, x in enumerate(text_file) if x == "$POS_X"]
    get_cluster_flag.append(len(text_file) + 1)
    return get_cluster_flag


# 格式化並插入群集資訊
def format_cluster_info_to_dict(
    unit_file_name,
    source_data,
    saved_data_type=None,
    structure_type=None,
    parent_name=None,
    parent_file_position=None,
    parent_clustered_string=None,
    x_y_clustered_string=None,
):
    Groups_info = []
    unit_file_path = f"./applications/{file}/GHSOM/output/{file}/{unit_file_name}.unit"
    print(unit_file_path)
    try:
        text_file = open(unit_file_path).read().split()
    except FileNotFoundError:
        print(f"Error: Unit file {unit_file_path} not found.")
        return Groups_info

    flag = get_cluster_flag(text_file)

    # 獲取層級索引
    layer_index = (
        int(unit_file_name.split("lvl")[1][0]) if "lvl" in unit_file_name else 1
    )

    XDIM = int(text_file[text_file.index("$XDIM") + 1])
    YDIM = int(text_file[text_file.index("$YDIM") + 1])
    map_size = XDIM * YDIM

    parent_name = (
        "Root" if parent_name is None else f"{parent_name}-{parent_file_position}"
    )
    x_y_clustered_string = "" if x_y_clustered_string is None else x_y_clustered_string
    parent_clustered_string = (
        "" if parent_clustered_string is None else parent_clustered_string
    )

    for i, map_index in zip(range(len(flag) - 1), range(map_size)):
        start_index, end_index = flag[i], flag[i + 1]
        currentSection = text_file[start_index:end_index]

        x_position = currentSection[currentSection.index("$POS_X") + 1]
        y_position = currentSection[currentSection.index("$POS_Y") + 1]

        group_position = f"{x_position}{y_position}"
        group_data_index = (
            currentSection[
                currentSection.index("$MAPPED_VECS")
                + 1 : currentSection.index("$MAPPED_VECS_DIST")
            ]
            if "$MAPPED_VECS" in currentSection
            else []
        )

        sub_map_file_name = (
            currentSection[currentSection.index("$URL_MAPPED_SOMS") + 1]
            if "$URL_MAPPED_SOMS" in currentSection
            else None
        )

        cluster_string = (
            f"{parent_clustered_string}{XDIM};{YDIM};{x_position};{y_position};"
        )
        x_y_string = f"{x_y_clustered_string}-{x_position}x{y_position}"

        index = np.array(group_data_index, dtype="int64")
        if index.ndim > 1:
            index = index.flatten()
        index = index.tolist()

        if sub_map_file_name:
            format_cluster_info_to_dict(
                sub_map_file_name,
                source_data,
                saved_data_type,
                structure_type,
                unit_file_name,
                group_position,
                cluster_string,
                x_y_string,
            )
        else:
            dimension_list = []
            for idx in index:
                if idx in df_source.index:
                    df_source.loc[idx, "clustered_label"] = cluster_string
                    df_source.loc[idx, "x_y_label"] = x_y_string

                    cluster_string = cluster_string.strip(";")
                    clusters_list = cluster_string.split(";")
                    levels = x_y_string.split("-")

                    for i in range(0, len(clusters_list), 4):
                        dimension_list.append(
                            [
                                clusters_list[i],
                                clusters_list[i + 1],
                                clusters_list[i + 2],
                                clusters_list[i + 3],
                            ]
                        )

                    for e in range(1, len(levels)):
                        df_source.loc[idx, f"clusterL{e}"] = levels[e]

                    point = GHSOM_center_point(dimension_list)
                    df_source.loc[idx, "point_x"] = point[0]
                    df_source.loc[idx, "point_y"] = point[1]
                else:
                    print(f"Warning: Index {idx} not found in df_source.")

    return Groups_info


# 計算中心點
def GHSOM_center_point(data_list):
    Bx = By = 1
    Bx_list, By_list, Point_list = [], [], []
    for item in data_list:
        Bx *= Fraction(1, int(item[0]))
        By *= Fraction(1, int(item[1]))
        Point = [Bx * int(item[2]), By * int(item[3])]
        Point_list.append(Point)
        Bx_list.append(Bx)
        By_list.append(By)

    Px = sum(point[0] for point in Point_list) + Bx_list[-1] / 2
    Py = sum(point[1] for point in Point_list) + By_list[-1] / 2
    return [Px, Py]


# 主程式執行
saved_file_type = "result_detail"
result = format_cluster_info_to_dict(prefix, df_source, saved_file_type, "flat")

output_dir = f"./applications/{file}/data"
os.makedirs(output_dir, exist_ok=True)
output_file = f"{output_dir}/{prefix}_with_clustered_label-{t1}-{t2}.csv"
df_source.to_csv(output_file, index=False)
print(f"Output saved to {output_file}")
