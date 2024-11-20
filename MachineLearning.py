# MachineLearning.py

import sys
sys.path.append("C:\\Users\\shian\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve, auc
import joblib
import jpholiday
import psutil
import os
from sklearn.cluster import DBSCAN
from pyproj import CRS
from astral import LocationInfo
from astral.sun import sun
from zoneinfo import ZoneInfo
exception_type, exception_object, exception_traceback = sys.exc_info()
import traceback
import pytz
from multiprocessing import Pool
from shapely.strtree import STRtree
# タイムゾーンの設定
JST = pytz.timezone('Asia/Tokyo')

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
    # タイムゾーンの生成
    #JST = timezone(timedelta(hours=+9), 'JST')
    s = sun(location.observer, date=date_time.date(), tzinfo=pytz.timezone('Asia/Tokyo'))
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

#ログ出力用
def print_samples(df, name, num_samples=5):
    print(f"\n{name} Samples:")
    # Print first `num_samples` samples
    print(f"First {num_samples} samples:")
    print(df[['latitude', 'longitude', '天候区分', '昼夜区分', 'weekday', 'is_holiday', 'road_shape', '発生日時']].head(num_samples))
    # Print random `num_samples` samples
    print(f"\nRandom {num_samples} samples:")
    print(df[['latitude', 'longitude', '天候区分', '昼夜区分', 'weekday', 'is_holiday', 'road_shape', '発生日時']].sample(num_samples, random_state=42))

# ネガティブポイント生成関数
def generate_random_points(polygon, num_points,accident_tree):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < num_points:
        remaining = num_points - len(points)
        # 一度に多くのポイントを生成
        random_x = np.random.uniform(minx, maxx, remaining * 2)
        random_y = np.random.uniform(miny, maxy, remaining * 2)
        candidate_points = gpd.GeoSeries([Point(x, y) for x, y in zip(random_x, random_y)])
        # ポリゴン内のポイントをフィルタリング
        within_polygon = candidate_points[candidate_points.within(polygon)]
        # 重複を排除
        unique_points = within_polygon[~within_polygon.isin(points)]
        # ポジティブポイントとの重複を排除
        unique_points = unique_points[~unique_points.apply(lambda p: is_duplicate(p, accident_tree))]
        points.extend(unique_points.tolist())
        if len(points) >= num_points:
            break
    return points[:num_points]

def is_duplicate(point, tree):
    possible_duplicates = tree.query(point)
    return any(point.equals(p) for p in possible_duplicates)

# クラスター処理関数（並列用）
def process_cluster(args):
    (road_width, road_alignment, road_shape, cluster_label, group, cluster_polygon, accident_tree, date_min_timestamp,date_max_timestamp, hour_distribution, minute_distribution, weekday_distribution, weather_distributions_per_month,weather_distribution_overall, positive_data_group) = args

    # ネガティブデータの数を設定
    num_neg_samples = len(group) * 3
    # 初期のネガティブポイントを一括生成
    negative_points = generate_random_points(cluster_polygon, num_neg_samples, accident_tree)

    if len(negative_points) < num_neg_samples:
        print(f"警告: クラスター {(road_width, road_alignment, road_shape, cluster_label)} のネガティブデータが目標数に達しませんでした。生成数: {len(negative_points)}")
        
    aeqd_crs = CRS(proj='aeqd', lat_0=35, lon_0=136, datum='WGS84', units='m')
    # GeoDataFrame作成
    neg_gdf = gpd.GeoDataFrame({'geometry': negative_points}, crs=aeqd_crs)
    # 座標系をWGS84に変換
    neg_gdf = neg_gdf.to_crs(epsg=4326)
    # 必要な情報を追加
    neg_gdf['longitude'] = neg_gdf.geometry.x
    neg_gdf['latitude'] = neg_gdf.geometry.y

    max_attempts = 10  # 再試行の最大回数
    attempts = 0
    
    # ネガティブデータの特徴量を一括で生成
    while attempts < max_attempts:
        # ネガティブデータの曜日をポジティブデータの分布に基づいてサンプリング
        sampled_weekdays = np.random.choice(
            weekday_distribution.index,
            size=num_neg_samples,
            p=weekday_distribution.values
        )

        # ランダムな日付を生成
        random_dates = pd.to_datetime(
            np.random.uniform(date_min_timestamp, date_max_timestamp, num_neg_samples),
            unit='s', utc=True
        ).tz_convert('Asia/Tokyo')

        # 曜日を調整
        # current_weekdays = random_dates.weekday
        # days_differences = (sampled_weekdays - current_weekdays) % 7
        # adjusted_dates = random_dates + pd.Timedelta(days=days_differences,unit='D')

        # ランダムな日付を対応する曜日に調整
        adjusted_dates = []
        for date, target_weekday in zip(random_dates, sampled_weekdays):
            current_weekday = date.weekday()
            days_difference = (target_weekday - current_weekday) % 7
            adjusted_date = date + pd.Timedelta(days=days_difference)
            adjusted_dates.append(adjusted_date)
        adjusted_dates = pd.to_datetime(adjusted_dates)


        # 時間と分をポジティブデータの分布からサンプリング
        neg_gdf['hour'] = np.random.choice(
            hour_distribution.index,
            size=len(neg_gdf),
            p=hour_distribution.values
        )
        neg_gdf['minute'] = np.random.choice(
            minute_distribution.index,
            size=len(neg_gdf),
            p=minute_distribution.values
        )

        # 発生日時を再構成
        neg_gdf['発生日時'] = adjusted_dates + pd.to_timedelta(neg_gdf['hour'], unit='h') + pd.to_timedelta(neg_gdf['minute'], unit='m')
        neg_gdf['発生日時'] = neg_gdf['発生日時'].dt.tz_convert('Asia/Tokyo')

        neg_gdf['month'] = neg_gdf['発生日時'].dt.month
        
        # 天候区分割り当て
        # 天候を月ごとの分布からサンプリング
        def assign_weather(row):
            month = row['month']
            weather_distribution = weather_distributions_per_month.get(month)
            if weather_distribution is not None and len(weather_distribution) > 0:
                return np.random.choice(
                    weather_distribution.index,
                    p=weather_distribution.values
                )
            else:
                # データがない場合は全体の分布からサンプリング
                return np.random.choice(
                    weather_distribution_overall.index,
                    p=weather_distribution_overall.values
                )
        
        neg_gdf['天候区分'] = neg_gdf.apply(assign_weather, axis=1)

        # 昼夜区分計算
        neg_gdf['昼夜区分'] = neg_gdf.apply(
            lambda row: get_day_night_code(row['latitude'], row['longitude'], row['発生日時']),
            axis=1
        )
        neg_gdf['昼夜区分'] = neg_gdf['昼夜区分'].apply(map_day_night)

        # その他の情報を追加
        neg_gdf['accident'] = 0  # ネガティブデータ
        neg_gdf['road_width'] = road_width
        neg_gdf['road_alignment'] = road_alignment
        neg_gdf['road_shape'] = road_shape
        neg_gdf['cluster'] = cluster_label
        neg_gdf['weekday'] = neg_gdf['発生日時'].dt.weekday
        neg_gdf['year'] = neg_gdf['発生日時'].dt.year
        neg_gdf['month'] = neg_gdf['発生日時'].dt.month
        neg_gdf['day'] = neg_gdf['発生日時'].dt.day
        neg_gdf['hour'] = neg_gdf['発生日時'].dt.hour
        neg_gdf['minute'] = neg_gdf['発生日時'].dt.minute
        neg_gdf['is_holiday'] = neg_gdf['発生日時'].apply(lambda x: jpholiday.is_holiday(x))

        # 重複チェック
        # ポジティブデータとの重複をチェック
        merged_positive = pd.merge(
            neg_gdf,
            positive_data_group,
            on=['latitude', 'longitude', 'month', 'day', 'hour', 'weekday', '天候区分'],
            how='inner'
        )

        # ネガティブデータ内での重複をチェック
        duplicated_negatives = neg_gdf.duplicated(subset=['latitude', 'longitude', 'month', 'day', 'hour', 'weekday', '天候区分'], keep=False)

        # 重複していないデータを抽出
        non_conflicting_negatives = neg_gdf[~neg_gdf.index.isin(merged_positive.index) & ~duplicated_negatives]

        # 重複しているデータを再生成
        conflicting_indices = neg_gdf.index.difference(non_conflicting_negatives.index)
        if len(conflicting_indices) == 0:
            # 重複がなくなったらループを抜ける
            neg_gdf = non_conflicting_negatives
            break
        else:
            # 再試行のために重複したデータのみを更新
            neg_gdf = neg_gdf.copy()
            neg_gdf.loc[conflicting_indices, '発生日時'] = None  # 発生日時をリセット
            attempts += 1

    if attempts >= max_attempts:
        print(f"警告: クラスター {(road_width, road_alignment, road_shape, cluster_label)} で重複のないネガティブデータを生成できませんでした。")

    return neg_gdf


if __name__ == "__main__":
    
    # 1. 事故データの読み込みと処理
    try:
        # 事故データの読み込み
        accident_data_2023 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2023.csv')
        accident_data_2022 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2022.csv')
        accident_data_2021 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2021.csv')
        accident_data_2020 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2020.csv')

        # データの結合
        #accident_data = pd.concat([accident_data_2023], ignore_index=True)
        # print(f"1年分のデータを使用します。")
        accident_data = pd.concat([accident_data_2023, accident_data_2022,accident_data_2021,accident_data_2020], ignore_index=True)
        print(f"4年分のデータを使用します。")
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

        # 列名のリネーム
        data = data.rename(columns={
            '発生日時　　年': 'year',
            '発生日時　　月': 'month',
            '発生日時　　日': 'day',
            '発生日時　　時': 'hour',
            '発生日時　　分': 'minute',
            '道路形状': 'road_shape',          # '道路形状' 列をリネーム
            '車道幅員': 'road_width',          # '車道幅員' 列をリネーム
            '道路線形': 'road_alignment'       # '道路線形' 列をリネーム
        })

        # 発生日時を datetime 型に変換
        data['発生日時'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])
        data = data.dropna(subset=['発生日時'])
        data['発生日時'] = data['発生日時'].dt.tz_localize('Asia/Tokyo')

        # 日時関連の特徴量を作成
        data['weekday'] = data['発生日時'].dt.weekday
        data['is_holiday'] = data['発生日時'].apply(lambda x: jpholiday.is_holiday(x))

        # 未知の天候を示すコードを定義
        UNKNOWN_WEATHER_CODE = 0
        UNKNOWN_DAY_NIGHT_CODE = 0

        # 欠損値を埋める
        data['天候'] = data['天候'].fillna(UNKNOWN_WEATHER_CODE)
        data['昼夜'] = data['昼夜'].fillna(UNKNOWN_DAY_NIGHT_CODE)
        data['road_shape'] = data['road_shape'].fillna('不明')
        data['road_width'] = data['road_width'].fillna('不明')
        data['road_alignment'] = data['road_alignment'].fillna('不明')

        data['天候区分'] = data['天候'].apply(map_weather)
        data['昼夜区分'] = data['昼夜'].apply(map_day_night)

        # ポジティブデータのサンプルを表示
        print_samples(data, "Positive Data")
        
        # 事故発生フラグを追加
        data['accident'] = 1

        # ポジティブデータのGeoDataFrameを作成
        data_positive_gdf = gpd.GeoDataFrame(
            data,
            geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
            crs='EPSG:4326'
        )

        # ポジティブデータから時間と分と曜日の分布を取得
        hour_distribution = data['hour'].value_counts(normalize=True)
        minute_distribution = data['minute'].value_counts(normalize=True)
        weekday_distribution = data['weekday'].value_counts(normalize=True)
        weather_distribution_overall = data['天候区分'].value_counts(normalize=True)


        # 月ごとの天候分布を取得
        weather_distributions_per_month = {}
        for month in data['month'].unique():
            month_data = data[data['month'] == month]
            weather_distribution = month_data['天候区分'].value_counts(normalize=True)
            weather_distributions_per_month[month] = weather_distribution

    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()

    # 2. クラスターの作成
    try:
        # 投影座標系への変換（距離計算のため）
        aeqd_crs = CRS(proj='aeqd', lat_0=35, lon_0=136, datum='WGS84', units='m')
        data_positive_gdf = data_positive_gdf.to_crs(aeqd_crs)

        # 'road_width'、'road_alignment'、'road_shape' ごとにデータを分割
        grouped = data_positive_gdf.groupby(['road_width', 'road_alignment', 'road_shape'])
        
        # 全てのクラスターを保持するリスト
        all_clusters = []
        cluster_polygons_list = []

        # 各グループごとにクラスタリングを実施
        for (road_width, road_alignment, road_shape), group in grouped:
            coords = np.array(list(zip(group.geometry.x, group.geometry.y)))

            # DBSCANによるクラスタリング（半径30メートル）
            db = DBSCAN(eps=30, min_samples=1).fit(coords)
            cluster_labels = db.labels_

            # クラスタリング結果をデータに追加
            group['cluster'] = cluster_labels

            # グループの情報を維持
            group['road_width'] = road_width
            group['road_alignment'] = road_alignment
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
                    cluster_polygon = cluster_geometry.convex_hull.buffer(30)  # ポリゴンを50メートル拡大

                cluster_polygons_list.append({
                    'road_width': road_width,
                    'road_alignment': road_alignment,
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
        print("geojson作成")

    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()

    # 3. ネガティブデータの生成
    try:
        # 日付の範囲を取得
        date_min, date_max = data['発生日時'].min(), data['発生日時'].max()
        # UTCに変換してタイムスタンプを取得
        date_min_timestamp = date_min.tz_convert('UTC').timestamp()
        date_max_timestamp = date_max.tz_convert('UTC').timestamp()
        
        # ポジティブデータの空間インデックス作成
        accident_points_list = list(clustered_data['geometry'])
        accident_tree = STRtree(accident_points_list)

        # クラスターごとに引数を準備
        args_list = []
        for (road_width, road_alignment, road_shape, cluster_label), group in clustered_data.groupby(['road_width', 'road_alignment', 'road_shape', 'cluster']):
            
            # クラスターのジオメトリを取得
            cluster_polygon = cluster_polygons_gdf[
                (cluster_polygons_gdf['road_width'] == road_width) &
                (cluster_polygons_gdf['road_alignment'] == road_alignment) &
                (cluster_polygons_gdf['road_shape'] == road_shape) &
                (cluster_polygons_gdf['cluster'] == cluster_label)
            ]['geometry'].iloc[0]
            
            args_list.append((road_width, road_alignment, road_shape, cluster_label, group, cluster_polygon, accident_tree,date_min_timestamp, date_max_timestamp, hour_distribution, minute_distribution, weekday_distribution,weather_distributions_per_month, weather_distribution_overall, group))

        # 並列処理でクラスタごとにネガティブデータ生成
        with Pool(processes=8) as pool:  # CPUコア数に応じて調整
            results = pool.map(process_cluster, args_list)

        # 結果を結合
        negative_data = pd.concat(results, ignore_index=True)
        
        # ネガティブデータのサンプルを表示
        print_samples(negative_data, "Negative Data")

    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()

    # 4. データの統合とモデルの作成
    try:
        # 座標系を元に戻す（WGS84）
        clustered_data = clustered_data.to_crs('EPSG:4326')
        negative_data = negative_data.to_crs('EPSG:4326')

        # 必要なカラムを選択
        features = ['latitude', 'longitude', 'month', 'day', 'hour', 'weekday', '昼夜区分', '天候区分', 'is_holiday', 'road_width', 'road_alignment', 'road_shape']
        target = 'accident'

        # カテゴリ変数のラベルエンコーディング
        label_encoders = {}
        for column in ['昼夜区分', '天候区分', 'road_width', 'road_alignment', 'road_shape']:
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
        traceback.print_exc()

    print("プログラムが完了しました。")