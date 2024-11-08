# evaluation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import joblib
import jpholiday
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from pyproj import CRS
from astral import LocationInfo
from astral.sun import sun
from multiprocessing import Pool
from shapely.strtree import STRtree

# フォントのパスを指定（お使いの環境に合わせてください）
font_path = 'C:\\Users\\ramiru\\AppData\\Local\\Microsoft\\Windows\\Fonts\\SourceHanSans-Medium.otf'
# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# フォント名を取得
font_name = font_prop.get_name()

# デフォルトフォントに設定
mpl.rcParams['font.family'] = font_name

# マイナス符号の文字化けを防ぐ
mpl.rcParams['axes.unicode_minus'] = False

# 緯度・経度の変換関数（既存の関数を使用）
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


def get_day_night_code(lat, lon, date_time):
    location = LocationInfo(latitude=lat, longitude=lon, timezone='Asia/Tokyo')
    # タイムゾーンの生成
    JST = timezone(timedelta(hours=+9), 'JST')
    s = sun(location.observer, date=date_time.date(), tzinfo=JST)
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
    road_shape, cluster_label, group, cluster_polygon, accident_tree, accident_points_list,weather_distribution_overall, date_min_timestamp, date_max_timestamp,hour_distribution,minute_distribution,weather_distributions_per_month,weekday_distribution = args
    
    # ネガティブデータの数を設定
    num_neg_samples = len(group) * 3
    negative_points = generate_random_points(cluster_polygon, num_neg_samples,accident_tree)

    if len(negative_points) < num_neg_samples:
        print(f"警告: クラスター {(road_shape, cluster_label)} のネガティブデータが目標数に達しませんでした。生成数: {len(negative_points)}")
        
    aeqd_crs = CRS(proj='aeqd', lat_0=35, lon_0=136, datum='WGS84', units='m')
    # GeoDataFrame作成
    neg_gdf = gpd.GeoDataFrame({'geometry': negative_points}, crs=aeqd_crs)
    # 座標系をWGS84に変換
    neg_gdf = neg_gdf.to_crs(epsg=4326)
    # 必要な情報を追加
    neg_gdf['longitude'] = neg_gdf.geometry.x
    neg_gdf['latitude'] = neg_gdf.geometry.y

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

    # '発生日時'を分単位に切り捨て
    neg_gdf['発生日時'] = neg_gdf['発生日時'].dt.floor('min')

    # 曜日を再計算
    neg_gdf['weekday'] = neg_gdf['発生日時'].dt.weekday

    # 月を取得
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

    # その他の特徴量追加
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

    return neg_gdf

if __name__ == "__main__":
    # 1. 2021年の事故データを読み込む
    accident_data_2019 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2019.csv')

    # 2. モデルで使用した前処理を同じように行う

    # 緯度・経度の欠損値を削除
    data = accident_data_2019.dropna(subset=['地点　緯度（北緯）', '地点　経度（東経）'])
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

    data['発生日時'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

    # 日時関連の特徴量を作成
    data['weekday'] = data['発生日時'].dt.weekday # 月曜日=0, 日曜日=6
    data['is_holiday'] = data['発生日時'].apply(lambda x: jpholiday.is_holiday(x))

    # 未知の天候を示すコードを定義
    UNKNOWN_WEATHER_CODE = 0
    UNKNOWN_DAY_NIGHT_CODE = 0

    # 欠損値を数値で埋める
    data['天候'] = data['天候'].fillna(UNKNOWN_WEATHER_CODE)
    data['昼夜'] = data['昼夜'].fillna(UNKNOWN_DAY_NIGHT_CODE)
    data['road_shape'] = data['road_shape'].fillna('不明')  # '道路形状' の欠損値を '不明' に置換

    # 昼夜区分の作成
    data['天候区分'] = data['天候'].apply(map_weather)
    data['昼夜区分'] = data['昼夜'].apply(map_day_night)

    # 事故発生フラグを追加
    data['accident'] = 1  # 事故が発生した

    # ポジティブデータのGeoDataFrameを作成
    data_positive_gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data['longitude'], data['latitude']),
        crs='EPSG:4326'
    )

    # 2. クラスターの読み込み
    cluster_polygons_gdf = gpd.read_file('cluster_polygons.geojson')

    # 4. 事故データにクラスター情報を割り当てる
    # 投影座標系への変換（クラスターポリゴンと一致させるため）
    aeqd_crs = CRS.from_proj4("+proj=aeqd +lat_0=35 +lon_0=136 +datum=WGS84 +units=m +no_defs")
    data_positive_gdf = data_positive_gdf.to_crs(aeqd_crs)
    cluster_polygons_gdf = cluster_polygons_gdf.to_crs(aeqd_crs)

    # クラスターポリゴンの空間インデックスを作成
    cluster_tree = STRtree(cluster_polygons_gdf.geometry)
    cluster_polygons_gdf['id'] = cluster_polygons_gdf.index

    # 各事故地点に対してクラスターポリゴンを検索
    assigned_clusters = []
    for idx, accident_point in data_positive_gdf.iterrows():
        point = accident_point.geometry
        road_shape = accident_point['road_shape']
        possible_clusters = cluster_polygons_gdf[cluster_polygons_gdf['road_shape'] == road_shape]
        possible_polygons = possible_clusters.geometry.tolist()
        matches = cluster_tree.query(point)
        found = False
        for polygon in matches:
            cluster_match = possible_clusters[possible_clusters.geometry == polygon]
            if not cluster_match.empty and polygon.contains(point):
                accident_point['cluster'] = cluster_match['cluster'].values[0]
                assigned_clusters.append(accident_point)
                found = True
                break
        if not found:
            # 該当するクラスターポリゴンがない場合でもデータを保持
            accident_point['cluster'] = -1  # クラスタが見つからない場合は -1 を設定
            assigned_clusters.append(accident_point)

    # クラスタ情報を持つ事故データのGeoDataFrameを作成
    assigned_data_gdf = gpd.GeoDataFrame(assigned_clusters, crs=aeqd_crs)

    # 5. ネガティブデータを生成（モデル訓練時と同じ方法）
    # 全体の分布から昼夜区分と天候区分の確率を取得
    day_night_distribution = data['昼夜区分'].value_counts(normalize=True)
    weather_distribution_overall = data['天候区分'].value_counts(normalize=True)
    weekday_distribution = data['weekday'].value_counts(normalize=True)
    hour_distribution = data['hour'].value_counts(normalize=True)
    minute_distribution = data['minute'].value_counts(normalize=True)

    # 月ごとの天候分布を取得
    weather_distributions_per_month = {}
    for month in data['month'].unique():
        month_data = data[data['month'] == month]
        weather_distribution = month_data['天候区分'].value_counts(normalize=True)
        weather_distributions_per_month[month] = weather_distribution

    # 日付の範囲を取得
    date_min, date_max = data['発生日時'].min(), data['発生日時'].max()
    date_min_timestamp = date_min.tz_convert('UTC').timestamp()
    date_max_timestamp = date_max.tz_convert('UTC').timestamp()

    # ネガティブデータを保持するリスト
    negative_samples_list = []

    # ポジティブデータの空間インデックス作成
    accident_points_list = list(assigned_data_gdf['geometry'])
    accident_tree = STRtree(accident_points_list)

    # クラスターごとに引数を準備
    args_list = []
    for (road_shape, cluster_label), group in assigned_data_gdf.groupby(['road_shape', 'cluster']):
        if cluster_label == -1:
            continue  # クラスタが見つからなかったデータはスキップ

        # クラスターのジオメトリを取得
        cluster_polygon = cluster_polygons_gdf[
            (cluster_polygons_gdf['road_shape'] == road_shape) & (cluster_polygons_gdf['cluster'] == cluster_label)
        ]['geometry'].iloc[0]

        args_list.append((
            road_shape, cluster_label, group, cluster_polygon, accident_tree, date_min_timestamp, date_max_timestamp,
            hour_distribution, minute_distribution, weekday_distribution, weather_distributions_per_month, weather_distribution_overall
        ))

    # 並列処理でネガティブデータを生成
    with Pool(processes=6) as pool:
        results = pool.map(process_cluster, args_list)

    # 全てのネガティブデータを結合
    negative_data = pd.concat(results, ignore_index=True)

    # 6. ポジティブデータとネガティブデータを結合
    # 必要なカラムを選択
    features = ['latitude', 'longitude', 'month', 'day', 'hour', 'minute', 'weekday', '昼夜区分', '天候区分', 'is_holiday','road_shape']
    target = 'accident'

    # データフレームを連結
    data_ml = pd.concat([assigned_data_gdf, negative_data], ignore_index=True)

    # 7. 座標系を元に戻す（WGS84）
    data_ml = data_ml.to_crs('EPSG:4326')
    
    # 'is_holiday'を数値に変換
    data_ml['is_holiday'] = data_ml['is_holiday'].astype(int)


    # 7. カテゴリ変数をエンコード（モデルと同じラベルエンコーダーを使用）
    # モデルで使用したラベルエンコーダーの読み込み
    label_encoders = joblib.load('label_encoders.pkl')

    for column in ['昼夜区分', '天候区分','road_shape']:
        le = label_encoders[column]
        data_ml[column] = le.transform(data_ml[column])

    # 6. 特徴量とターゲットを分割
    X_test = data_ml[features]
    y_test = data_ml[target]

    # 7. モデルの読み込み
    model = joblib.load('accident_risk_model.pkl')

    # 8. 予測の実行
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]  # 予測確率

    # 9. 評価指標の計算
    # 日本語で表示するためにラベルを設定
    target_names = ['事故なし (0)', '事故あり (1)']
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0)
    print(report)
    print(confusion_matrix(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_probs)
    print(f"AUC スコア: {auc_score:.4f}")

    # 10. 各カテゴリ別のエラー分析

    # 10.1 天候コードから元のラベルに戻す
    inverse_weather_dict = {index: label for index, label in enumerate(label_encoders['天候区分'].classes_)}
    data_ml['天候区分_label'] = data_ml['天候区分'].map(inverse_weather_dict)

    # 10.2 昼夜コードから元のラベルに戻す
    inverse_day_night_dict = {index: label for index, label in enumerate(label_encoders['昼夜区分'].classes_)}
    data_ml['昼夜区分_label'] = data_ml['昼夜区分'].map(inverse_day_night_dict)

    # 10.3 月を抽出（既に作成済み）

    # カテゴリ別に指標を計算する関数（日本語のラベルを使用）
    def calculate_metrics(grouped_data, group_name):
        metrics = []
        for name, group in grouped_data:
            idx = group.index
            y_true = y_test.loc[idx]
            y_pred_group = y_pred[idx]
            # サンプル数が0の場合はスキップ
            if len(y_true) == 0:
                continue
            # クラスの種類を確認
            if len(np.unique(y_true)) < 2:
                # 片方のクラスしかない場合、適合率などは計算できないので NaN を代入
                precision = recall = f1 = np.nan
                accuracy = accuracy_score(y_true, y_pred_group)
            else:
                # 評価指標を計算
                precision = precision_score(y_true, y_pred_group, zero_division=0)
                recall = recall_score(y_true, y_pred_group, zero_division=0)
                f1 = f1_score(y_true, y_pred_group, zero_division=0)
                accuracy = accuracy_score(y_true, y_pred_group)
            metrics.append({
                group_name: name,
                '正解率': accuracy,
                '適合率': precision,
                '再現率': recall,
                'F1スコア': f1
            })
        metrics_df = pd.DataFrame(metrics)
        return metrics_df

    # 10.4 天候別の分析
    weather_groups = data_ml.groupby('天候区分_label')
    weather_metrics_df = calculate_metrics(weather_groups, '天候')

    # print("天候別の結果:")
    # print("weather_metrics_df.columns:", weather_metrics_df.columns)
    # print("weather_metrics_df:")
    # print(weather_metrics_df)

    # 10.5 月別の分析
    month_groups = data_ml.groupby('month')
    month_metrics_df = calculate_metrics(month_groups, '月')

    # 10.6 時間別の分析
    hour_groups = data_ml.groupby('hour')
    hour_metrics_df = calculate_metrics(hour_groups, '時間')

    # print("hour_metrics_df.columns:", hour_metrics_df.columns)
    # print("hour_metrics_df:")
    # print(hour_metrics_df)

    # 10.7 昼夜別の分析
    day_night_groups = data_ml.groupby('昼夜区分_label')
    day_night_metrics_df = calculate_metrics(day_night_groups, '昼夜区分')

    # print("day_night_metrics_df.columns:", day_night_metrics_df.columns)
    # print("day_night_metrics_df:")
    # print(day_night_metrics_df)

    # 10.8 曜日別の分析
    weekday_groups = data_ml.groupby('weekday')
    weekday_metrics_df = calculate_metrics(weekday_groups, '曜日')

    # print("weekday_metrics_df.columns:", weekday_metrics_df.columns)
    # print("weekday_metrics_df:")
    # print(weekday_metrics_df)

    # 曜日を日本語に変換
    weekday_labels = ['月', '火', '水', '木', '金', '土', '日']
    weekday_metrics_df['曜日_label'] = weekday_metrics_df['曜日'].apply(lambda x: weekday_labels[int(x)])

    # 11. 結果の可視化

    metrics = ['正解率', '適合率', '再現率', 'F1スコア']

    # 天候別の結果をコマンドプロンプトに表示
    print("天候別の結果:")
    print(weather_metrics_df)

    # 月別の結果をコマンドプロンプトに表示
    print("月別の結果:")
    print(month_metrics_df)

    # 時間別の結果をコマンドプロンプトに表示
    print("時間別の結果:")
    print(hour_metrics_df)

    # 昼夜別の結果をコマンドプロンプトに表示
    print("昼夜別の結果:")
    print(day_night_metrics_df)

    # 曜日別の結果をコマンドプロンプトに表示
    print("曜日別の結果:")
    print(weekday_metrics_df)

    # 14. 誤分類の詳細分析（オプション）
    # 誤分類データの抽出
    false_negatives = data_ml[(y_test == 1) & (y_pred == 0)]
    false_positives = data_ml[(y_test == 0) & (y_pred == 1)]

    # 誤分類データの数を表示
    print(f"False Negatives（偽陰性）の数: {len(false_negatives)}")
    print(f"False Positives（偽陽性）の数: {len(false_positives)}")

    # 11.1 天候別のグラフ
    weather_metrics_df.set_index('天候')[metrics].plot(kind='bar', figsize=(10, 6))
    plt.title('天候別のモデル性能')
    plt.ylabel('スコア')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(loc='lower right')
    plt.savefig('weather.png')
    plt.show()

    # 11.2 月別のグラフ
    if '月' in month_metrics_df.columns:
        month_metrics_df = month_metrics_df.sort_values('月')
        month_metrics_df.set_index('月')[metrics].plot(kind='bar', figsize=(12, 6))
        plt.title('月別のモデル性能')
        plt.ylabel('スコア')
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.savefig('month.png')
        plt.show()
    else:
        print("月別のデータがありません。")

    # 11.3 時間別のグラフ
    if '時間' in hour_metrics_df.columns:
        hour_metrics_df = hour_metrics_df.sort_values('時間')
        hour_metrics_df.set_index('時間')[metrics].plot(kind='bar', figsize=(15, 6))
        plt.title('時間別のモデル性能')
        plt.ylabel('スコア')
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.savefig('time.png')
        plt.show()
    else:
        print("時間別のデータがありません。")

    # 11.4 昼夜別のグラフ
    if '昼夜区分' in day_night_metrics_df.columns:
        day_night_metrics_df.set_index('昼夜区分')[metrics].plot(kind='bar', figsize=(10, 6))
        plt.title('昼夜別のモデル性能')
        plt.ylabel('スコア')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.savefig('daynight.png')
        plt.show()
    else:
        print("昼夜区分別のデータがありません。")

    # 11.5 曜日別のグラフ
    if '曜日_label' in weekday_metrics_df.columns:
        weekday_metrics_df = weekday_metrics_df.sort_values('曜日')
        weekday_metrics_df.set_index('曜日_label')[metrics].plot(kind='bar', figsize=(10, 6))
        plt.title('曜日別のモデル性能')
        plt.ylabel('スコア')
        plt.xticks(rotation=0)
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.savefig('weekday.png')
        plt.show()
    else:
        print("曜日別のデータがありません。")

    # 12. 誤分類の詳細分析（オプション）

    # 誤分類データの抽出
    false_negatives = data_ml[(y_test == 1) & (y_pred == 0)]
    false_positives = data_ml[(y_test == 0) & (y_pred == 1)]

    # 誤分類データの数を表示
    print(f"False Negatives（偽陰性）の数: {len(false_negatives)}")
    print(f"False Positives（偽陽性）の数: {len(false_positives)}")
