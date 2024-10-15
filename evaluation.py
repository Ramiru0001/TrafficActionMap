# evaluation.py

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import jpholiday
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
import matplotlib.font_manager as fm
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# 1. 2021年の事故データを読み込む
accident_data_2022 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2021.csv')

# 2. モデルで使用した前処理を同じように行う

# 緯度・経度の欠損値を削除
data = accident_data_2022.dropna(subset=['地点　緯度（北緯）', '地点　経度（東経）'])

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

# 緯度・経度の変換
data['latitude'] = data['地点　緯度（北緯）'].apply(dms_str_to_dd)
data['longitude'] = data['地点　経度（東経）'].apply(dms_str_to_dd)

# 発生日時をdatetime型に変換
data = data.rename(columns={
    '発生日時　　年': 'year',
    '発生日時　　月': 'month',
    '発生日時　　日': 'day',
    '発生日時　　時': 'hour',
    '発生日時　　分': 'minute'
})

data['発生日時'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

# 年、月、日、曜日、時間帯などの特徴量を作成
data['year'] = data['発生日時'].dt.year
data['month'] = data['発生日時'].dt.month
data['day'] = data['発生日時'].dt.day
data['hour'] = data['発生日時'].dt.hour
data['minute'] = data['発生日時'].dt.minute
data['weekday'] = data['発生日時'].dt.weekday  # 月曜日=0, 日曜日=6

# 祝日情報の追加
data['is_holiday'] = data['発生日時'].apply(lambda x: jpholiday.is_holiday(x))

# 未知の天候を示すコードを定義
UNKNOWN_WEATHER_CODE = 0
UNKNOWN_DAY_NIGHT_CODE = 0

# 欠損値を数値で埋める
data['天候'] = data['天候'].fillna(UNKNOWN_WEATHER_CODE)
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

# 事故発生フラグを追加
data['accident'] = 1  # 事故が発生した

# 必要なカラムを選択
features = ['latitude', 'longitude', 'hour', 'weekday', '昼夜区分', '天候区分', 'is_holiday']
target = 'accident'

data_positive = data[features + [target]]

# 3. ネガティブデータを生成（モデル訓練時と同じ方法）

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

# ネガティブデータの特徴量を選択
neg_samples = neg_samples[features + ['accident', 'month', '発生日時']]

# 4. データを結合
data_ml = pd.concat([data_positive, neg_samples], ignore_index=True)

# 5. カテゴリ変数をエンコード（モデルと同じラベルエンコーダーを使用）

# モデルで使用したラベルエンコーダーの読み込み
label_encoders = joblib.load('label_encoders.pkl')

for column in ['昼夜区分', '天候区分']:
    le = label_encoders[column]
    data_ml[column] = le.transform(data_ml[column])

# 'is_holiday'を数値に変換
data_ml['is_holiday'] = data_ml['is_holiday'].astype(int)

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
