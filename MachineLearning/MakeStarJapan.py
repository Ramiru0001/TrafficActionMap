import sys
sys.path.append("C:\\Users\\shian\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages")
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib as mpl
import traceback
# ---- 事故データの読み込み ----
# 例: 2023年分のみを仮定 (実ファイル名・パスは調整)
accident_data_2023 = pd.read_csv("AccidentData/honhyo_2023.csv")
# データの結合
accident_data = pd.concat([accident_data_2023], ignore_index=True)
# ---- 緯度/経度列の確認・変換 ----
# ここでは機械学習用コードで使われるdms_str_to_ddのような変換を行っていると仮定
# 例として `地点　緯度（北緯）` と `地点　経度（東経）` を degree へ変換済みだと想定します。
# もし dms形式なら、適宜 `machinelearning.py`のdms_str_to_dd関数などを使用してください。

# カラム名例: 'latitude', 'longitude' が既にあるとして進めます
# もし実際は "地点　緯度（北緯）" などの場合、以下のようにrenameします:
# accident_data_2023.rename(columns={
#     "地点　緯度（北緯）": "latitude",
#     "地点　経度（東経）": "longitude"
# }, inplace=True)

# 緯度・経度の欠損値を削除
data = accident_data.dropna(subset=['地点　緯度（北緯）', '地点　経度（東経）'])

# 緯度・経度の変換関数
def dms_str_to_dd(dms_str):
    try:
        dms_str = str(dms_str).zfill(10)
        if len(dms_str) == 9:  # 緯度の場合
            degrees = int(dms_str[0:2])
            minutes = int(dms_str[2:4])
            seconds = int(dms_str[4:6])
            fraction = int(dms_str[6:9]) / 1000
        elif len(dms_str) == 10:  # 経度の場合
            degrees = int(dms_str[0:3])
            minutes = int(dms_str[3:5])
            seconds = int(dms_str[5:7])
            fraction = int(dms_str[7:10]) / 1000
        else:
            return None

        seconds = seconds + fraction
        dd = degrees + minutes / 60 + seconds / 3600
        return dd
    
    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()

# 緯度・経度の変換
data['latitude'] = data['地点　緯度（北緯）'].apply(dms_str_to_dd)
data['longitude'] = data['地点　経度（東経）'].apply(dms_str_to_dd)


# ---- 座標を任意の刻みで丸めてグループ化 ----
# 例: 小数点第2位まで丸め（約1km四方）
data['lat_bin'] = data['latitude'].round(2)
data['lon_bin'] = data['longitude'].round(2)

# 同じbinに属する事故を集計
counts = data.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')

# GeoDataFrame化
gdf = gpd.GeoDataFrame(
    counts,
    geometry=gpd.points_from_xy(counts["lon_bin"], counts["lat_bin"]),
    crs="EPSG:4326"
)

# ---- Webメルカトル (EPSG:3857) へ変換 ----
gdf_3857 = gdf.to_crs(epsg=3857)

# ---- 夜空風の描画 ----
fig, ax = plt.subplots(figsize=(12, 12))
# 背景を夜空のような紫がかった黒に
fig.patch.set_facecolor("#2c003e")
ax.set_facecolor("#2c003e")

# 軸や枠線、グリッドをOFF
ax.set_axis_off()
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.box(False)

# 事故数countに応じて星を大きく + カラフル
# お好みで調整 (例えば最大値が大きすぎるならもっと縮める等)
scaleFactor = 0.05
baseSize = 30
sizes = baseSize + gdf_3857["count"] * scaleFactor

# 星形マーカーでカラフルに
# 例: cmap="plasma" (カラフル) / "YlOrBr" (金系) / "hot" (赤系) など試してみてください
sc = ax.scatter(
    gdf_3857.geometry.x,
    gdf_3857.geometry.y,
    s=sizes,
    c=gdf_3857["count"],         # 事故数に応じて色変化
    cmap="YlOrBr",                # カラーマップ
    alpha=0.8,
    marker="*",                   # 星形
    edgecolor="none"
)
# 表示範囲設定: データ全体を少し余裕を持って含む
minx, miny, maxx, maxy = gdf_3857.total_bounds
xrange = maxx - minx
yrange = maxy - miny
ax.set_xlim(minx - xrange*0.05, maxx + xrange*0.05)
ax.set_ylim(miny - yrange*0.05, maxy + yrange*0.05)

# 余白最小化
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)

# 保存
out_file = "accident_star_night_grouped.png"
plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor())
plt.close(fig)

print(f"座標をまとめて星空風に描画した '{out_file}' を生成しました。")