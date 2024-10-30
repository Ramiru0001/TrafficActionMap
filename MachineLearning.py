# MachineLearning.py

import sys
sys.path.append("C:\\Users\\shian\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages")
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import jpholiday
import psutil
import os
from sklearn.cluster import DBSCAN
from pyproj import CRS
from astral import LocationInfo
from astral.sun import sun

# メモリエラー対策
max_memory_usage_mb = 40000  # 最大メモリ使用量をMB単位で設定

def check_memory_usage(max_usage_mb):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / (1024 * 1024)  # メモリ使用量をMB単位で取得
    if mem_usage_mb > max_usage_mb:
        raise MemoryError(f"メモリ使用量が {max_usage_mb} MB を超えました。")

def get_day_night_code(lat, lon, date_time):
    location = LocationInfo(latitude=lat, longitude=lon, timezone='Asia/Tokyo')
    s = sun(location.observer, date=date_time.date(), tzinfo=jst)
    sunrise = s['sunrise']
    sunset = s['sunset']
    
    # 日の出・日の入り時刻の前後1時間を計算
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
        return None  # 不明
    
# 1. 事故データの読み込みと処理
try:
    # 事故データの読み込み
    accident_data_2023 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2023.csv')
    accident_data_2022 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2022.csv')

    # データの結合
    accident_data = pd.concat([accident_data_2023, accident_data_2022], ignore_index=True)

    # 緯度・経度の欠損値を削除
    data = accident_data.dropna(subset=['地点　緯度（北緯）', '地点　経度（東経）'])

    # 緯度・経度の変換関数
    def dms_str_to_dd(dms_str):
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

    # 緯度・経度の変換
    data['latitude'] = data['地点　緯度（北緯）'].apply(dms_str_to_dd)
    data['longitude'] = data['地点　経度（東経）'].apply(dms_str_to_dd)

    # 列名のリネーム
    data = data.rename(columns={
        '発生日時　　年': 'year',
        '発生日時　　月': 'month',
        '発生日時　　日': 'day',
        '発生日時　　時': 'hour',
        '発生日時　　分': 'minute',
        '道路形状': 'road_shape'  # '道路形状' 列をリネーム
    })

    # 発生日時を datetime 型に変換
    data['発生日時'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

    # 日時関連の特徴量を作成
    data['weekday'] = data['発生日時'].dt.weekday
    data['is_holiday'] = data['発生日時'].apply(lambda x: jpholiday.is_holiday(x))

    # 未知の天候を示すコードを定義
    UNKNOWN_WEATHER_CODE = 0
    UNKNOWN_DAY_NIGHT_CODE = 0

    # 欠損値を埋める
    data['天候'] = data['天候'].fillna(UNKNOWN_WEATHER_CODE)
    data['昼夜'] = data['昼夜'].fillna(UNKNOWN_DAY_NIGHT_CODE)
    data['road_shape'] = data['road_shape'].fillna('不明')  # '道路形状' の欠損値を '不明' に置換

    # 天候と昼夜の区分をマッピング
    def map_weather(code):
        weather_dict = {
            0: '不明', 1: '晴', 2: '曇', 3: '雨', 4: '霧', 5: '雪'
        }
        return weather_dict.get(int(code), '不明')

    def map_day_night(code):
        day_night_dict = {
            0: '不明', 11: '昼_明', 12: '昼_昼', 13: '昼_暮',
            21: '夜_暮', 22: '夜_夜', 23: '夜_明'
        }
        return day_night_dict.get(int(code), '不明')

    data['天候区分'] = data['天候'].apply(map_weather)
    data['昼夜区分'] = data['昼夜'].apply(map_day_night)

    # 事故発生フラグを追加
    data['accident'] = 1

    # ポジティブデータのGeoDataFrameを作成
    data_positive_gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
        crs='EPSG:4326'
    )

except Exception as e:
    print(f"エラー: {e}")

# 2. クラスターの作成
try:
    # 投影座標系への変換（距離計算のため）
    aeqd_crs = CRS(proj='aeqd', lat_0=35, lon_0=136, datum='WGS84', units='m')
    data_positive_gdf = data_positive_gdf.to_crs(aeqd_crs)

    # '道路形状'ごとにデータを分割
    road_shape_groups = data_positive_gdf.groupby('road_shape')

    # 全てのクラスターを保持するリスト
    all_clusters = []
    cluster_polygons_list = []

    # 各 '道路形状' グループごとにクラスタリングを実施
    for road_shape, group in road_shape_groups:
        coords = np.array(list(zip(group.geometry.x, group.geometry.y)))

        # DBSCANによるクラスタリング（半径30メートル）
        db = DBSCAN(eps=30, min_samples=1).fit(coords)
        cluster_labels = db.labels_

        # クラスタリング結果をデータに追加
        group['cluster'] = cluster_labels

        # 'road_shape' を維持
        group['road_shape'] = road_shape

        # クラスターをリストに追加
        all_clusters.append(group)

        # クラスターごとにポリゴンを作成
        for cluster_label in np.unique(cluster_labels):
            cluster_group = group[group['cluster'] == cluster_label]
            cluster_geometry = cluster_group.geometry.union_all()

            # クラスターのジオメトリがポイントの場合、バッファを作成
            if cluster_geometry.geom_type == 'Point':
                cluster_polygon = cluster_geometry.buffer(30)  # 15メートルのバッファ
            else:
                cluster_polygon = cluster_geometry.convex_hull.buffer(50)  # ポリゴンを50メートル拡大

            cluster_polygons_list.append({
                'road_shape': road_shape,
                'cluster': cluster_label,
                'geometry': cluster_polygon
            })

    # 全てのクラスターを結合
    clustered_data = pd.concat(all_clusters, ignore_index=True)
    
    # クラスターポリゴンのGeoDataFrameを作成
    cluster_polygons_gdf = gpd.GeoDataFrame(cluster_polygons_list, crs=aeqd_crs)

    # クラスターポリゴンを保存（後で予測時に使用するため）
    cluster_polygons_gdf.to_crs('EPSG:4326').to_file('cluster_polygons.geojson', driver='GeoJSON')

except Exception as e:
    print(f"エラー: {e}")

# 3. ネガティブデータの生成
try:
    # 全体の分布から昼夜区分と天候区分の確率を取得
    weather_distribution = data['天候区分'].value_counts(normalize=True)

    # 日付の範囲を取得
    date_min, date_max = data['発生日時'].min(), data['発生日時'].max()
    date_min_timestamp = date_min.value / 1e9  # ナノ秒を秒に変換
    date_max_timestamp = date_max.value / 1e9

    # ネガティブデータを保持するリスト
    negative_samples_list = []

    # 各クラスターごとにネガティブデータを生成
    for (road_shape, cluster_label), group in clustered_data.groupby(['road_shape', 'cluster']):
        # クラスター内の事故地点を取得
        accident_points = group['geometry'].values

        # クラスターのジオメトリを取得
        cluster_polygon = cluster_polygons_gdf[
            (cluster_polygons_gdf['road_shape'] == road_shape) & (cluster_polygons_gdf['cluster'] == cluster_label)
        ]['geometry'].iloc[0]

        minx, miny, maxx, maxy = cluster_polygon.bounds

        # ネガティブデータの数を設定（ポジティブデータの数と同じにする）
        num_neg_samples = len(group)* 3

        negative_points = []
        attempts = 0
        max_attempts = num_neg_samples * 10

        while len(negative_points) < num_neg_samples and attempts < max_attempts:
            random_point = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if cluster_polygon.contains(random_point) and not any(random_point.equals(p) for p in accident_points):
                negative_points.append(random_point)
            attempts += 1

        if len(negative_points) < num_neg_samples:
            print(f"警告: クラスター {(road_shape, cluster_label)} のネガティブデータが目標数に達しませんでした。生成数: {len(negative_points)}")

        # ネガティブデータのGeoDataFrameを作成
        neg_gdf = gpd.GeoDataFrame(
            {'geometry': negative_points},
            crs=aeqd_crs
        )

        # 必要な情報を追加
        neg_gdf['longitude'] = neg_gdf.geometry.x
        neg_gdf['latitude'] = neg_gdf.geometry.y

        # 発生日時をランダムに生成
        random_timestamps = np.random.uniform(date_min_timestamp, date_max_timestamp, len(neg_gdf))
        neg_gdf['発生日時'] = pd.to_datetime(random_timestamps, unit='s')

        neg_gdf['昼夜区分'] = get_day_night_code(neg_gdf['longitude'], neg_gdf['latitude'], data['発生日時'])

         # 天候区分を全体の分布に基づいて割り当て
        neg_gdf['天候区分'] = np.random.choice(
            weather_distribution.index,
            size=len(neg_gdf),
            p=weather_distribution.values
        )

        # その他の情報を追加
        neg_gdf['weekday'] = neg_gdf['発生日時'].dt.weekday
        neg_gdf['year'] = neg_gdf['発生日時'].dt.year
        neg_gdf['month'] = neg_gdf['発生日時'].dt.month
        neg_gdf['day'] = neg_gdf['発生日時'].dt.day
        neg_gdf['hour'] = neg_gdf['発生日時'].dt.hour
        neg_gdf['minute'] = neg_gdf['発生日時'].dt.minute
        neg_gdf['is_holiday'] = neg_gdf['発生日時'].apply(lambda x: jpholiday.is_holiday(x))
        neg_gdf['accident'] = 0  # ネガティブデータ
        neg_gdf['road_shape'] = road_shape
        neg_gdf['cluster'] = cluster_label

        negative_samples_list.append(neg_gdf)

    # 全てのネガティブデータを結合
    negative_data = pd.concat(negative_samples_list, ignore_index=True)

except Exception as e:
    print(f"エラー: {e}")

# 4. データの統合とモデルの作成
try:
    # 座標系を元に戻す（WGS84）
    clustered_data = clustered_data.to_crs('EPSG:4326')
    negative_data = negative_data.to_crs('EPSG:4326')

    # 必要なカラムを選択
    features = ['latitude', 'longitude', 'month', 'day', 'hour', 'minute','weekday', '昼夜区分', '天候区分', 'is_holiday','road_shape']
    target = 'accident'

    # カテゴリ変数のラベルエンコーディング
    label_encoders = {}
    for column in ['昼夜区分', '天候区分','road_shape']:
        le = LabelEncoder()
        combined_values = pd.concat([clustered_data[column], negative_data[column]], ignore_index=True).unique()
        le.fit(combined_values)
        clustered_data[column] = le.transform(clustered_data[column])
        negative_data[column] = le.transform(negative_data[column])
        label_encoders[column] = le

    # データを結合
    data_ml = pd.concat([clustered_data, negative_data], ignore_index=True)

    # 'is_holiday'を数値に変換
    data_ml['is_holiday'] = data_ml['is_holiday'].astype(int)

    # 特徴量とターゲットの設定
    X = data_ml[features]
    y = data_ml[target]

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルの定義
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )

    # モデルの訓練
    model.fit(X_train, y_train)

    # テストデータでの予測
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 評価指標の計算
    print(f"評価結果:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba)}")
    print(confusion_matrix(y_test, y_pred))

    # モデルの保存
    model_filename = f'accident_risk_model.pkl'
    joblib.dump(model, model_filename)

    # ラベルエンコーダーの保存
    encoder_filename = f'label_encoders.pkl'
    joblib.dump(label_encoders, encoder_filename)

except Exception as e:
    print(f"エラー: {e}")

print("プログラムがが完了しました。")