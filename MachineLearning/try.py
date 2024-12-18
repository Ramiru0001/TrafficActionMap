import sys
sys.path.append("C:\\Users\\shian\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages")
import pandas as pd
import numpy as np
import datetime
import pytz
import jpholiday
import joblib
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from shapely.geometry import Point
import matplotlib as mpl
import contextily as ctx
from shapely.geometry import box

# フォントのパスを指定（お使いの環境に合わせてください）
font_path = './Fonts//NotoSansJP-Light.ttf'
# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# フォント名を取得
font_name = font_prop.get_name()

# デフォルトフォントに設定
mpl.rcParams['font.family'] = font_name

# マイナス符号の文字化けを防ぐ
mpl.rcParams['axes.unicode_minus'] = False

# 昼夜区分と天候区分をmapする関数 (get_risk_data参照)
def map_day_night(code):
    day_night_dict = {
        0: '不明', 11: '昼_明', 12: '昼_昼', 13: '昼_暮',
        21: '夜_暮', 22: '夜_夜', 23: '夜_明'
    }
    return day_night_dict.get(int(code), '不明')

def get_day_night_code(lat, lon, date_time):
    # 厳密な日の出・日の入り計算があるならばimport astral等必要
    # ここでは簡略化し、固定値にするか、get_risk_data同等のロジックを適用してください。
    # 実際に同じ処理を行いたい場合は、MachineLearning.pyやproject.pyで定義済みのget_day_night_codeを使用。
    from astral import LocationInfo
    from astral.sun import sun
    jst = pytz.timezone('Asia/Tokyo')
    location = LocationInfo(latitude=lat, longitude=lon, timezone='Asia/Tokyo')
    s = sun(location.observer, date=date_time.date(), tzinfo=jst)
    sunrise = s['sunrise']
    sunset = s['sunset']
    sunrise_minus_1h = sunrise - pd.Timedelta(hours=1)
    sunrise_plus_1h = sunrise + pd.Timedelta(hours=1)
    sunset_minus_1h = sunset - pd.Timedelta(hours=1)
    sunset_plus_1h = sunset + pd.Timedelta(hours=1)

    if sunrise <= date_time < sunrise_plus_1h:
        return 11  # 昼－明
    elif sunrise_plus_1h <= date_time < sunset_minus_1h:
        return 12  # 昼－昼
    elif sunset_minus_1h <= date_time < sunset:
        return 13  # 昼－暮
    elif sunset <= date_time < sunset_plus_1h:
        return 21  # 夜－暮
    elif sunset_plus_1h <= date_time or date_time < sunrise_minus_1h:
        return 22  # 夜－夜
    elif sunrise_minus_1h <= date_time < sunrise:
        return 23  # 夜－明
    else:
        return 0  # 不明


    
#############################################
# 必要なデータのパス設定
model_path = "../output_data/accident_risk_model.pkl"
label_encoders_path = "../output_data/label_encoders.pkl"
cluster_polygons_path = "../output_data/cluster_polygons.geojson"
#############################################

# モデル・エンコーダ読込
model = joblib.load(model_path)
label_encoders = joblib.load(label_encoders_path)

# クラスターシェープ読み込み
cluster_polygons_gdf = gpd.read_file(cluster_polygons_path)
cluster_polygons_gdf = cluster_polygons_gdf.to_crs(epsg=4326)

# 日本全土予測日時・天候を固定
jst = pytz.timezone('Asia/Tokyo')
date_time = jst.localize(datetime.datetime(2024, 7, 15, 12, 0))  # 例：2024年7月15日12時
weather_str = "晴"

# cluster_polygons_gdfから特徴量作成
input_data_list = []
for idx, cluster_row in cluster_polygons_gdf.iterrows():
    cluster_centroid = cluster_row['geometry'].centroid
    cluster_lat = cluster_centroid.y
    cluster_lon = cluster_centroid.x
    road_width = cluster_row['road_width']
    road_alignment = cluster_row['road_alignment']
    road_shape = cluster_row['road_shape']

    # 特徴量計算
    is_holiday = int(jpholiday.is_holiday(date_time.date()))
    month = date_time.month
    day = date_time.day
    hour = date_time.hour
    minute = date_time.minute
    weekday = date_time.weekday()
    day_night_code = get_day_night_code(cluster_lat, cluster_lon, date_time)
    day_night = map_day_night(day_night_code)

    input_data = {
        'latitude': cluster_lat,
        'longitude': cluster_lon,
        'month': month,
        'day': day,
        'hour': hour,
        'minute': minute,
        'weekday': weekday,
        'is_holiday': is_holiday,
        '昼夜区分': day_night,
        '天候区分': weather_str,
        'road_width': road_width,
        'road_alignment': road_alignment,
        'road_shape': road_shape,
        'cluster': cluster_row['cluster']
    }
    input_data_list.append(input_data)

input_df = pd.DataFrame(input_data_list)

# カテゴリ変数のエンコード
for column in ['昼夜区分', '天候区分', 'road_width', 'road_alignment', 'road_shape']:
    le = label_encoders[column]
    input_df[column] = le.transform(input_df[column])

# 特徴量カラム（学習時と同じ）
feature_columns = ['latitude', 'longitude', 'month', 'day', 'hour', 'weekday', '昼夜区分', '天候区分', 'is_holiday', 'road_width', 'road_alignment', 'road_shape']
X = input_df[feature_columns]

# 予測
risk_scores = model.predict_proba(X)[:,1]

input_df['accident'] = risk_scores

# クラスター単位で集約（平均）
risk_summary = input_df.groupby('cluster').agg({
    'latitude':'first',
    'longitude':'first',
    'accident':'mean'
}).reset_index()

# GeoDataFrameを作り、背景地図（contextily）用に投影を変更
points_gdf = gpd.GeoDataFrame(
    risk_summary,
    geometry=gpd.points_from_xy(risk_summary['longitude'], risk_summary['latitude']),
    crs='EPSG:4326'
)

# contextilyで描画するにはWebメルカトル(EPSG:3857)へ
points_gdf_3857 = points_gdf.to_crs(epsg=3857)

# 日本全体の範囲 (北海道から九州)
japan_bounds = [122.93457, 24.396308, 153.986672, 45.551483]

# 日本の範囲をポリゴンで定義
geometry = box(*japan_bounds)
japan_gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")

# 投影法を Web Mercator に変換
japan_gdf = japan_gdf.to_crs(epsg=3857)

# 描画
fig, ax = plt.subplots(figsize=(12, 12))
japan_gdf.plot(ax=ax, edgecolor="red", facecolor="none")  # 日本の範囲を赤枠で表示

try:
    ctx.add_basemap(
        ax, 
        crs=japan_gdf.crs, 
        source=ctx.providers.OpenStreetMap.Mapnik,  # カラフルな地図スタイル
        zoom=10  # 解像度を上げる
    )

    # リスク散布
    sc = ax.scatter(
        points_gdf_3857.geometry.x,
        points_gdf_3857.geometry.y,
        c=points_gdf_3857['accident'],
        cmap='Reds',
        alpha=0.7,
        s=10
    )

    # タイトルと調整
    ax.set_title("日本全体の地図 (カラフルなデザイン)", fontsize=16)
    ax.set_axis_off()
    

    plt.show()
    plt.savefig("japan_cluster_accident_risk_poster_with_map.png", dpi=300)
    plt.close(fig)

except Exception as e:
    print(f"ベースマップの追加に失敗しました: {e}")
# cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
# cbar.set_label('Accident Risk Probability', fontsize=10)

# ax.set_title("Predicted Accident Risk Across Japan (Clusters)", fontsize=12)
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")

# plt.tight_layout()
# plt.savefig("japan_cluster_accident_risk_poster_with_map.png", dpi=300)
# plt.close(fig)

# print("A4ポスター 'japan_cluster_accident_risk_poster_with_map.png' が背景地図付きで作成されました。")