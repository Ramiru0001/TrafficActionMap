# app.py
import sys
sys.path.append("C:\\Users\\shian\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages")
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from flask import Flask, render_template, request, jsonify
import requests
import xml.etree.ElementTree as ET
import datetime
import jpholiday
from geopy.distance import geodesic
import joblib
from astral import LocationInfo
from astral.sun import sun
import numpy as np
import os  # 環境変数から API キーを取得するために追加
# OpenWeatherMap用のモジュールをインポート
import get_OpenWeatherMap
import traceback
import math
import pytz

app = Flask(__name__)
# 環境変数から API キーを取得
api_key = os.environ.get('OPENWEATHERMAP_API_KEY')

# モデルとラベルエンコーダーの読み込み
model = joblib.load('accident_risk_model_date.pkl')
label_encoders = joblib.load('label_encoders_date.pkl')

# 日本標準時のタイムゾーンを取得
jst = pytz.timezone('Asia/Tokyo')

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

def is_holiday():
    today = datetime.date.today()
    return jpholiday.is_holiday(today)

def get_weather(lat, lon,api_key):
    try:
        # get_OpenWeatherMap.py の get_current_weather 関数を呼び出す
        weather_info = get_OpenWeatherMap.get_current_weather(lat, lon, api_key)
        if weather_info:
            #print(f"取得した天気情報: {weather_info}")
            return weather_info
        else:
            print("天気情報の取得に失敗しました。")
            return None
    except Exception as e:
        traceback.print_exc()  # エラーの詳細を表示
        return None
def extract_weather_info(data, class20_code):
    # データ内の timeSeries を走査
    print(f"デバッグ: 指定された class20_code = {class20_code}")
    for series in data[0]['timeSeries']:
        # 'areas' を走査
        for area in series['areas']:
            # area の code が class20_code と一致する場合
            if area['area']['code'] == class20_code:
                # 天気情報を取得
                if 'weathers' in area:
                    return area['weathers'][0]  # 最初の天気情報を返す
                elif 'weatherCodes' in area:
                    # weatherCodes から天気情報を取得する場合（必要に応じて実装）
                    pass
    return None  # 該当する天気情報が見つからない場合

def get_weather_code(weather):
    if weather=="晴":
        return 1
    elif weather=="曇":
        return 2
    elif weather=="雨":
        return 3
    elif weather=="霧":
        return 4
    elif weather=="雪":
        return 5
    else:
        return None  # 不明

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_weather_and_risk_data', methods=['POST'])
def get_weather_and_risk_data():
    try:
        data = request.get_json()
        lat = data['lat']
        lon = data['lon']
        weather_data = get_weather(lat,lon,api_key)
        weather=weather_data['weather_type'] 
        #print(f"緯度 {lat}、経度 {lon} の天気は: {weather} です。",flush=True)
        if not weather:
            return jsonify({'error': '天気データの取得に失敗しました'}), 500

        # 仮のデータを返す
        #weather = '晴れ'
        risk_data = []

        return jsonify({'weather': weather, 'riskData': risk_data})
    except Exception as e:
        traceback.print_exc()  # スタックトレースをコンソールに出力
        return jsonify({'error': 'サーバーエラーが発生しました'}), 500

@app.route('/get_risk_data', methods=['POST'])
def get_risk_data():
    data = request.get_json()
    weather = data['weather']
    datetime_str = data['datetime']
    latitude = data['latitude']
    longitude = data['longitude']
    prediction_duration = data['prediction_duration']#予想時間(分)
    prediction_radius=data['prediction_radius']#予想半径(m)

    # 地球の半径を6371kmとする
    radius_km = prediction_radius / 1000.0

    # 緯度1度あたり約111km
    lat_step = 0.00001  # 9桁の緯度
    lon_step = 0.00001  # 10桁の経度（近似）

    # 範囲内の緯度経度を生成
    points = []
    lat_min = latitude - radius_km / 111.0
    lat_max = latitude + radius_km / 111.0
    lon_min = longitude - radius_km / (111.0 * math.cos(math.radians(latitude)))
    lon_max = longitude + radius_km / (111.0 * math.cos(math.radians(latitude)))

    current_lat = lat_min
    while current_lat <= lat_max:
        current_lon = lon_min
        while current_lon <= lon_max:
            points.append({'latitude': round(current_lat, 9), 'longitude': round(current_lon, 10)})
            current_lon += lon_step
        current_lat += lat_step
        
    # 緯度・経度の範囲を日本全体に設定
    # lat_min = 24.0  # 日本の南端
    # lat_max = 46.0  # 日本の北端
    # lon_min = 123.0  # 日本の西端
    # lon_max = 146.0  # 日本の東端

    # datetime_str を datetime オブジェクトに変換
    date_time = datetime.datetime.fromisoformat(datetime_str)
    date_time = jst.localize(date_time)  # 日本時間に設定

    # is_holiday をサーバー側で判定
    is_holiday = int(jpholiday.is_holiday(date_time.date()))
    
    month = date_time.month
    day = date_time.day
    hour = date_time.hour
    minute = date_time.minute
    weekday = date_time.weekday()

    # 昼夜区分の計算
    day_night = get_day_night_code(latitude, longitude, date_time)
    #天候区分の計算
    weather=get_weather_code(weather)

     # 入力データフレームを作成
    input_data = pd.DataFrame({
        'latitude': latitude,
        'longitude': longitude,
        'month': month,
        'day': day,
        'hour': hour,
        'minute': minute,
        'weekday': weekday,
        'is_holiday': is_holiday,
        '昼夜区分': day_night,
        '天候区分': weather
    }) 
    
    # カテゴリ変数のエンコード
    for column in ['昼夜区分', '天候区分']:
        le = label_encoders[column]
        input_data[column] = le.transform(input_data[column])

    # 'is_holiday'を数値に変換
    input_data['is_holiday'] = input_data['is_holiday'].astype(int)

    # フィーチャーの順序を訓練時と一致させる
    feature_columns = ['latitude', 'longitude', 'month', 'day', 'hour', 'minute',  'weekday', '昼夜区分', '天候区分', 'is_holiday']
    input_data = input_data[feature_columns]

    # 事故リスクの予測
    risk_scores = model.predict_proba(input_data)[:, 1]
    input_data['risk_score'] = risk_scores

    # リスクスコアが高い地点をフィルタリング（任意）
    high_risk_data = input_data[input_data['risk_score'] > 0.5]

    # 必要な情報を辞書形式で返す
    risk_data = high_risk_data[['latitude', 'longitude', 'risk_score']].to_dict(orient='records')
    
    # 結果をクライアントに返す
    return jsonify({
        'riskData': risk_data
        # その他のデータ
    })

def some_risk_prediction_function(lat, lon, weather, datetime):
    # 実際のリスク予測ロジックを実装
    # ダミーのリスクスコア
    return 0.5

if __name__ == '__main__':
    app.run(debug=True)
