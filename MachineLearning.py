#MachineLearning.py
import pandas as pd
import numpy as np
from datetime import datetime
# import folium
# from folium.plugins import HeatMap
#import matplotlib.pyplot as plt
#import matplotlib.font_manager as fm
#import matplotlib as mpl
from sklearn.model_selection import train_test_split


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


# 事故データの読み込み
#accident_data = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2023.csv')

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

# 地図の中心を設定
# avg_lat = data['latitude'].mean()
# avg_lon = data['longitude'].mean()
# m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)

# ヒートマップ用のデータを準備
#heat_data = data[['latitude', 'longitude']].values.tolist()

# ヒートマップを追加
#HeatMap(heat_data, radius=8, blur=4, max_zoom=13).add_to(m)

# # マーカーを追加（必要に応じて）
# for index, row in data.iterrows():
#     folium.CircleMarker(
#         location=[row['latitude'], row['longitude']],
#         radius=1,
#         color='red',
#         fill=True,
#         fill_color='red',
#         fill_opacity=0.6
#     ).add_to(m)

# 地図を保存
#m.save('accident_heatmap.html')

# 時間帯別の事故件数を集計
hour_counts = data['hour'].value_counts().sort_index()

# 事故発生フラグを追加
data['accident'] = 1

# データの範囲を取得
lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
date_min, date_max = data['発生日時'].min(), data['発生日時'].max()

# ランダムにネガティブデータを生成
num_samples = len(data)
np.random.seed(42)

neg_samples = pd.DataFrame({
    'latitude': np.random.uniform(lat_min, lat_max, num_samples),
    'longitude': np.random.uniform(lon_min, lon_max, num_samples),
    '発生日時': pd.to_datetime(np.random.uniform(date_min.value, date_max.value, num_samples))
})

# データから昼夜区分の分布を取得
day_night_distribution = data['昼夜区分'].value_counts(normalize=True)

# ネガティブデータに昼夜区分を割り当て
neg_samples['昼夜区分'] = np.random.choice(
    day_night_distribution.index,
    size=num_samples,
    p=day_night_distribution.values
)

# データから天候の分布を取得
weather_distribution = data['天候区分'].value_counts(normalize=True)

# ネガティブデータに天候を割り当て
neg_samples['天候区分'] = np.random.choice(
    weather_distribution.index,
    size=num_samples,
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

# 必要なカラムを選択
features = ['latitude', 'longitude', 'hour', 'weekday', '昼夜区分', '天候区分', 'is_holiday']
target = 'accident'  # 事故の有無（後述）

# ポジティブデータの特徴量を選択
data_positive = data[features + [target]]

# ネガティブデータの特徴量を選択
neg_samples = neg_samples[features + [target]]

# データを結合
data_ml = pd.concat([data_positive, neg_samples], ignore_index=True)

from sklearn.preprocessing import LabelEncoder

# カテゴリ変数をラベルエンコーディング
label_encoders = {}
for column in ['昼夜区分', '天候区分']:
    le = LabelEncoder()
    data_ml[column] = le.fit_transform(data_ml[column])
    label_encoders[column] = le

# 'is_holiday'を数値に変換
data_ml['is_holiday'] = data_ml['is_holiday'].astype(int)

#2
X = data_ml[features]
y = data_ml[target]

from sklearn.model_selection import train_test_split

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# モデルの定義
model = RandomForestClassifier(n_estimators=100, random_state=42)

# モデルの訓練
model.fit(X_train, y_train)

# テストデータでの予測
y_pred = model.predict(X_test)

# 評価指標の計算
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

import joblib

# モデルの保存
joblib.dump(model, 'accident_risk_model.pkl')

# ラベルエンコーダーの保存
joblib.dump(label_encoders, 'label_encoders.pkl')
