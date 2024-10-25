# check_proximity_duplicates.py

import pandas as pd
import numpy as np
import math

# Haversine 関数の定義
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # 地球の半径（メートル）
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(delta_lambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

# 事故データの読み込み
accident_data_2023 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2023.csv')
accident_data_2022 = pd.read_csv('TrafficAccidentMap_Data/read_codeChange/honhyo_2022.csv')

# データの結合
accident_data = pd.concat([accident_data_2023, accident_data_2022], ignore_index=True)

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
        return np.nan  # 不正なフォーマットの場合

    # 秒に小数点以下を追加
    seconds = seconds + fraction

    # 十進法に変換
    dd = degrees + minutes / 60 + seconds / 3600
    return dd

# 緯度・経度の変換
accident_data['latitude'] = accident_data['地点　緯度（北緯）'].apply(dms_str_to_dd)
accident_data['longitude'] = accident_data['地点　経度（東経）'].apply(dms_str_to_dd)

# 欠損値を削除
data = accident_data.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

# 10メートル以内の事故をカウント
count_nearby = 0
total = len(data)

for i in range(total):
    lat1 = data.at[i, 'latitude']
    lon1 = data.at[i, 'longitude']
    for j in range(i+1, total):
        lat2 = data.at[j, 'latitude']
        lon2 = data.at[j, 'longitude']
        distance = haversine(lat1, lon1, lat2, lon2)
        if distance <= 10:
            count_nearby += 2  # iとjの両方をカウント
            break  # 一つでも近接が見つかれば次のiへ

percentage = (count_nearby / total) * 100

print("=== 緯度・経度が10メートル以内で発生している事故の割合 ===")
print(f"総データ数: {total}")
print(f"10メートル以内で発生している事故数: {count_nearby}")
print(f"割合: {percentage:.2f}%")