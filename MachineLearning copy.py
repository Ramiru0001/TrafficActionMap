#MachineLearning.py
import sys
sys.path.append("C:\\Users\\shian\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages")
import pandas as pd
import numpy as np
from datetime import datetime
# import folium
# from folium.plugins import HeatMap
#import matplotlib.pyplot as plt
#import matplotlib.font_manager as fm
#import matplotlib as mpl
from sklearn.model_selection import train_test_split
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import jpholiday
import psutil
import os

# # フォントのパスを指定（お使いの環境に合わせてください）
# font_path = 'C:\\Users\\ramiru\\AppData\\Local\\Microsoft\\Windows\\Fonts\\SourceHanSans-Medium.otf'
# # フォントプロパティを作成
# font_prop = fm.FontProperties(fname=font_path)

# # フォント名を取得
# font_name = font_prop.get_name()

# # デフォルトフォントに設定
# mpl.rcParams['font.family'] = font_name

# # マイナス符号の文字化けを防ぐ
# mpl.rcParams['axes.unicode_minus'] = False

max_memory_usage_mb = 40000  # 最大メモリ使用量をMB単位で設定（例：1GB）
#メモリエラー
def check_memory_usage(max_usage_mb):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_usage_mb = mem_info.rss / (1024 * 1024)  # メモリ使用量をMB単位で取得
    if mem_usage_mb > max_usage_mb:
        raise MemoryError(f"メモリ使用量が {max_usage_mb} MB を超えました。")

    
# 1. 事故データの読み込みと処理（ポジティブデータ）
try:

    # 事故データの読み込み
    accident_data_2023 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2023.csv')
    accident_data_2022 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2022.csv')

    # 2つのデータを結合
    accident_data = pd.concat([accident_data_2023, accident_data_2022], ignore_index=True)

    # 2. 緯度・経度の欠損値を削除
    data = accident_data.dropna(subset=['地点　緯度（北緯）', '地点　経度（東経）'])

    # 緯度・経度の変換関数
    def dms_str_to_dd(dms_str):
        dms_str = str(dms_str).zfill(10)  # ゼロ埋めして10桁にする
        if len(dms_str) == 9:  # 緯度の場合
            degrees = int(dms_str[0:2])
            minutes = int(dms_str[2:4])
            seconds = int(dms_str[4:6])
            fraction = int(dms_str[6:9]) / 1000  # 小数点以下の秒
        elif len(dms_str) == 10:  # 経度の場合
            degrees = int(dms_str[0:3])
            minutes = int(dms_str[3:5])
            seconds = int(dms_str[5:7])
            fraction = int(dms_str[7:10]) / 1000  # 小数点以下の秒
        else:
            return None  # 不正なフォーマットの場合

        # 秒に小数点以下を追加
        seconds = seconds + fraction

        # 十進法に変換
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
        '発生日時　　分': 'minute'
    })

    # 発生日時を datetime 型に変換
    data['発生日時'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

    # 年、月、日、曜日、時間帯などの特徴量を作成
    data['year'] = data['発生日時'].dt.year
    data['month'] = data['発生日時'].dt.month
    data['day'] = data['発生日時'].dt.day
    data['hour'] = data['発生日時'].dt.hour
    data['minute'] = data['発生日時'].dt.minute
    data['weekday'] = data['発生日時'].dt.weekday  # 月曜日=0, 日曜日=6

    # 祝日情報の追加
    import jpholiday

    data['is_holiday'] = data['発生日時'].apply(lambda x: jpholiday.is_holiday(x))

    # 未知の天候を示すコードを定義
    UNKNOWN_WEATHER_CODE = 0
    # 未知の昼夜を示すコードを定義
    UNKNOWN_DAY_NIGHT_CODE = 0

    # 欠損値を数値で埋める
    data['天候'] = data['天候'].fillna(UNKNOWN_WEATHER_CODE)
    # 欠損値を数値で埋める
    data['昼夜'] = data['昼夜'].fillna(UNKNOWN_DAY_NIGHT_CODE)

    # 天候の処理
    def map_weather(code):
        weather_dict = {
            0: '不明', 1: '晴', 2: '曇', 3: '雨', 4: '霧', 5: '雪'
        }
        return weather_dict.get(int(code), '不明')

    # 昼夜のコードを区分にマッピングする関数
    def map_day_night(code):
        day_night_dict = {
            0: '不明', 11: '昼_明', 12: '昼_昼', 13: '昼_暮',
            21: '夜_暮', 22: '夜_夜', 23: '夜_明'
        }
        return day_night_dict.get(int(code), '不明')

    # 昼夜区分の作成
    data['天候区分'] = data['天候'].apply(map_weather)
    data['昼夜区分'] = data['昼夜'].apply(map_day_night)

    # 時間帯別の事故件数を集計
    hour_counts = data['hour'].value_counts().sort_index()

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
    
# 2. 日本の陸地をエリアに分割
try:
    # Natural Earthのシェープファイルのパス
    natural_earth_shp = 'ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'  # ダウンロードしたシェープファイルのパスに置き換えてください

    # シェープファイルを読み込む
    world = gpd.read_file(natural_earth_shp)

    # 日本のポリゴンを抽出
    japan = world[world['ADMIN'] == 'Japan']

    # CRSを確認・設定（WGS84: EPSG:4326）
    japan = japan.to_crs(epsg=4326)

    # 日本の陸地をカバーするグリッドを生成

    # 日本の緯度経度の範囲（おおよそ）
    LAT_MIN, LAT_MAX = 24.0, 46.0
    LON_MIN, LON_MAX = 122.0, 146.0

    # 分割するグリッドの行数と列数を計算（目標のポリゴン数に近づける）
    # 例えば、縦10行、横10列で100個のグリッドを作成
    NUM_ROWS = 30
    NUM_COLS = 30

    # 緯度・経度の間隔を計算
    lat_steps = np.linspace(LAT_MIN, LAT_MAX, NUM_ROWS + 1)
    lon_steps = np.linspace(LON_MIN, LON_MAX, NUM_COLS + 1)

    # グリッドセル（ポリゴン）を生成
    grid_polygons = []
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            # 四隅の座標を取得
            lon_min = lon_steps[j]
            lon_max = lon_steps[j + 1]
            lat_min = lat_steps[i]
            lat_max = lat_steps[i + 1]
            # ポリゴンを作成
            polygon = Polygon([
                (lon_min, lat_min),
                (lon_max, lat_min),
                (lon_max, lat_max),
                (lon_min, lat_max)
            ])
            grid_polygons.append(polygon)

    # グリッドをGeoDataFrameに変換
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_polygons}, crs='EPSG:4326')

    # グリッドと日本の陸地ポリゴンを重ね合わせ、陸地部分のみを抽出
    japan_grids = gpd.overlay(grid_gdf, japan, how='intersection')

    # 各グリッドにエリアIDを割り当て
    japan_grids['area_id'] = japan_grids.reset_index().index

    # エリアポリゴンを保存
    japan_grids.to_file('japan_area_grids.geojson', driver='GeoJSON')

    # エリア数を確認
    num_areas = len(japan_grids)
    print(f"生成されたエリア数: {num_areas}")

    # ポジティブデータにエリアIDを割り当て
    data_positive = gpd.sjoin(data_positive_gdf, japan_grids[['geometry', 'area_id']], how='inner', predicate='within')


    # 必要なカラムを選択
    features = ['latitude', 'longitude', 'month', 'day', 'hour', 'minute', 'weekday', '昼夜区分', '天候区分', 'is_holiday']
    target = 'accident'

    # data_positive = data_positive[features + [target, 'area_id']]

    # # ポジティブデータの特徴量を選択
    # data_positive = data[features + [target]]
    
except Exception as e:
    print(f"エラー: {e}")

# 3. ラベルエンコーダーの準備
try:
    # カテゴリ変数のクラスを定義
    day_night_categories = ['不明', '昼_明', '昼_昼', '昼_暮', '夜_暮', '夜_夜', '夜_明']
    weather_categories = ['不明', '晴', '曇', '雨', '霧', '雪']

    label_encoders = {}
    for column, categories in [('昼夜区分', day_night_categories), ('天候区分', weather_categories)]:
        le = LabelEncoder()
        le.fit(categories)
        label_encoders[column] = le

    # 4. 全体の分布から昼夜区分と天候区分の確率を取得

    day_night_distribution = data['昼夜区分'].value_counts(normalize=True)
    weather_distribution = data['天候区分'].value_counts(normalize=True)
    
except Exception as e:
    print(f"エラー: {e}")
    
# 5. エリアごとのモデルの作成

date_min, date_max = data['発生日時'].min(), data['発生日時'].max()

# date_min と date_max を秒単位のタイムスタンプに変換
date_min_timestamp = date_min.value / 1e9  # ナノ秒を秒に変換
date_max_timestamp = date_max.value / 1e9

for idx, row in japan_grids.iterrows():
    area_id = row['area_id']
    polygon = row['geometry']
    
    try:
        # メモリ使用量をチェック
        check_memory_usage(max_memory_usage_mb)
        
        print(f"エリア {area_id} の処理を開始します。")
        
        # エリア内のポジティブデータを取得
        area_positive = data_positive[data_positive['area_id'] == area_id]
        
        if area_positive.empty:
            print(f"エリア {area_id} にはポジティブデータが存在しません。スキップします。")
            continue
        
        # エリア内のネガティブデータを生成
        
        # ポリゴンのバウンディングボックスを取得
        minx, miny, maxx, maxy = polygon.bounds
        
        # 緯度方向のステップ（約10メートル）
        lat_step = 10 / 111000  # 約0.00009度
        
        # 緯度の配列を生成
        latitudes = np.arange(miny, maxy, lat_step)
        
        points = []
        
        for lat in latitudes:
            cos_lat = np.cos(np.radians(lat))
            # 経度方向のステップ（約10メートル）
            lon_step = 10 / (111000 * cos_lat)
            longitudes = np.arange(minx, maxx, lon_step)
            for lon in longitudes:
                # 前後左右5メートルの範囲でランダムにずらす
                rand_lat_shift = np.random.uniform(-5, 5) / 111000
                rand_lon_shift = np.random.uniform(-5, 5) / (111000 * cos_lat)
                shifted_lat = lat + rand_lat_shift
                shifted_lon = lon + rand_lon_shift
                point = Point(shifted_lon, shifted_lat)
                if polygon.contains(point):
                    points.append((shifted_lat, shifted_lon))

        if not points:
            print(f"エリア {area_id} でネガティブポイントが生成されませんでした。スキップします。")
            continue
        
        # メモリ使用量をチェック
        check_memory_usage(max_memory_usage_mb)
        
    except Exception as e:
        print(f"エラー: {e}")
    
    except MemoryError as me:
        print(f"メモリエラーが発生しました: {me}")
        print("処理を中断します。")
        break  # ループを抜けてプログラムを終了
    
    
    try:
        # ネガティブデータのDataFrameを作成
        neg_samples = pd.DataFrame(points, columns=['latitude', 'longitude'])
        num_neg_samples = len(neg_samples)
        
        # 発生日時をランダムに生成
        random_timestamps = np.random.uniform(date_min_timestamp, date_max_timestamp, num_neg_samples)
        neg_samples['発生日時'] = pd.to_datetime(random_timestamps, unit='s')

        # 昼夜区分と天候区分を全体の分布に基づいて割り当て
        neg_samples['昼夜区分'] = np.random.choice(
            day_night_distribution.index,
            size=num_neg_samples,
            p=day_night_distribution.values
        )
        neg_samples['天候区分'] = np.random.choice(
            weather_distribution.index,
            size=num_neg_samples,
            p=weather_distribution.values
        )


        # 特徴量の作成
        neg_samples['year'] = neg_samples['発生日時'].dt.year
        neg_samples['month'] = neg_samples['発生日時'].dt.month
        neg_samples['day'] = neg_samples['発生日時'].dt.day
        neg_samples['hour'] = neg_samples['発生日時'].dt.hour
        neg_samples['minute'] = neg_samples['発生日時'].dt.minute
        neg_samples['weekday'] = neg_samples['発生日時'].dt.weekday
        neg_samples['is_holiday'] = neg_samples['発生日時'].apply(lambda x: jpholiday.is_holiday(x))
        neg_samples['accident'] = 0  # 事故が発生しなかったフラグ

        # ネガティブデータとポジティブデータを結合
        area_positive = area_positive[features + [target]]
        neg_samples = neg_samples[features + [target]]
        
        # メモリ使用量をチェック
        check_memory_usage(max_memory_usage_mb)
    
        # マージ用の特徴量を指定（'accident' 列を除く）
        merge_features = features.copy()

        # 重複を削除するために特徴量でマージ（左側がネガティブ、右側がポジティブ）
        merged = neg_samples.merge(
            area_positive[merge_features],
            on=merge_features,
            how='left',
            indicator=True
        )

        # 'left_only' はネガティブサンプルのみを意味する
        neg_samples_unique = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        
        # 必要な列を再度追加（'accident' 列を0で設定）
        neg_samples_unique['accident'] = 0

        print(f"ポジティブサンプルの総数: {len(area_positive)}")
        print(f"ネガティブサンプルの総数: {len(neg_samples)}")
        print(f"重複削除後のネガティブサンプル数: {len(neg_samples_unique)}")

        # データを結合
        area_data = pd.concat([area_positive, neg_samples_unique], ignore_index=True)
        
        # 'accident' 列に欠損値がないことを確認
        if area_data['accident'].isnull().any():
            print(f"警告: 'accident' 列に欠損値が含まれています。欠損値を0で埋めます。:{area_data.isnull().sum()}")
            area_data['accident'] = area_data['accident'].fillna(0)

    except MemoryError as me:
        print(f"メモリエラーが発生しました: {me}")
        print("処理を中断します。")
        break  # ループを抜けてプログラムを終了
    
    except Exception as e:
        print(f"エラー: {e}")

    try:
        # カテゴリ変数をラベルエンコーディング
        for column in ['昼夜区分', '天候区分']:
            le = label_encoders[column]
            area_data[column] = le.transform(area_data[column])
            
        # 'is_holiday'を数値に変換
        area_data['is_holiday'] = area_data['is_holiday'].astype(int)
        
        # 特徴量とターゲットの設定
        X = area_data[features]
        y = area_data[target]
        
        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # モデルの定義
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # モデルの訓練
        model.fit(X_train, y_train)
        
        # テストデータでの予測
        y_pred = model.predict(X_test)
        
        # 評価指標の計算
        print(f"エリア {area_id} の評価結果:")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        
        # モデルの保存
        model_filename = f'accident_risk_model_area_{area_id}.pkl'
        joblib.dump(model, model_filename)
        print(f"エリア{area_id}モデルの作成に成功しました。")
        # ラベルエンコーダーの保存
        encoder_filename = f'label_encoders_area_{area_id}.pkl'
        joblib.dump(label_encoders, encoder_filename)
        print(f"エリア{area_id}モデルのラベルエンコーダーの作成に成功しました。")

        # メモリ解放
        del area_data, X, y, X_train, X_test, y_train, y_test, y_pred, model
        
    except Exception as e:
        print(f"エラー: {e}")
        
print("全てのエリアのモデル作成が完了しました。")
