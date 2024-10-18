# app.py
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
import pytz
from get_weather_area_code import get_weather_area_code, get_office_code_from_class20_code

import traceback
app = Flask(__name__)

# モデルとラベルエンコーダーの読み込み
model = joblib.load('accident_risk_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# 予測用の関数
def predict_accident_risk(input_data):
    # カテゴリ変数のエンコード
    for column in ['昼夜区分', '天候区分']:
        le = label_encoders[column]
        input_data[column] = le.transform(input_data[column])
    
    # 予測確率の取得
    accident_proba = model.predict_proba(input_data)[:, 1][0]
    return accident_proba

#日の出・日の入り時刻の前後1時間を計算します。
#入力の日時date_timeがどの時間帯に属するかを判定し、対応する「昼夜」コードを返します。
import pytz

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

# エンドポイントの作成

@app.route('/predict_accident_risk', methods=['POST'])
def predict_accident_risk_endpoint():
    data = request.get_json()
    lat = data['lat']
    lon = data['lon']
    hour = data['hour']
    datetime_str = data['datetime']  # クライアントから日時を取得
    weekday = data['weekday']
    is_holiday = data['is_holiday']
    day_night = data['day_night']  # 昼夜区分
    weather = data['weather']      # 天候区分

    # 日時文字列をdatetime型に変換
    date_time = pd.to_datetime(datetime_str)

    # 祝日判定
    is_holiday = int(jpholiday.is_holiday(date_time))

    # 昼夜コードを計算
    day_night_code = get_day_night_code(lat, lon, date_time)

    if day_night_code is None:
        return jsonify({'error': '昼夜コードの計算に失敗しました'}), 500
    
    input_data = pd.DataFrame({
        'latitude': [lat],
        'longitude': [lon],
        'hour': [hour],
        'weekday': [weekday],
        'is_holiday': [is_holiday],
        '昼夜区分': [day_night],
        '天候区分': [weather]
    })
    # フィーチャーの順序を訓練時と一致させる
    feature_columns = ['latitude', 'longitude', 'hour', 'weekday', 'is_holiday', '昼夜区分', '天候区分']
    input_data = input_data[feature_columns]

    risk_score = predict_accident_risk(input_data)

    return jsonify({'riskScore': risk_score})

def is_holiday():
    today = datetime.date.today()
    return jpholiday.is_holiday(today)

def get_weather(lat, lon):
    try:
        # 緯度・経度から class20s のコードを取得
        class20_code = get_weather_area_code(lat, lon)
        if not class20_code:
            print("地域コードの取得に失敗しました。")
            return None
        
        # class20s コードから office コードを取得
        office_code = get_office_code_from_class20_code(class20_code)
        if not office_code:
            print("office コードの取得に失敗しました。")
            return None
        
        # 天気データの取得
        url  = f'https://www.jma.go.jp/bosai/forecast/data/overview_forecast/{office_code}.json'
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # 天気情報を解析して、指定した class20_code に対応するエリアの天気を取得
            weather_info = extract_weather_info(data, class20_code)
            if weather_info:
                print(f"取得した天気情報: {weather_info}")
                return weather_info
            else:
                print("天気情報の解析に失敗しました。")
                return None

        except Exception as e:
            print(f"天気データの取得に失敗しました。エラー: {e}")
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_weather_and_risk_data', methods=['POST'])
def get_weather_and_risk_data():
    try:
        data = request.get_json()
        lat = data['lat']
        lon = data['lon']
        weather = get_weather(lat,lon)
        print(f"緯度 {lat}、経度 {lon} の天気は: {weather} です。",flush=True)
        if not weather:
            return jsonify({'error': '天気データの取得に失敗しました'}), 500

        # 仮のデータを返す
        #weather = '晴れ'
        risk_data = []

        return jsonify({'weather': weather, 'riskData': risk_data})
    except Exception as e:
        traceback.print_exc()  # スタックトレースをコンソールに出力
        return jsonify({'error': 'サーバーエラーが発生しました'}), 500

# app.py の更新部分

@app.route('/get_risk_data', methods=['POST'])
def get_risk_data():
    data = request.get_json()
    weather = data['weather']
    hour = data['hour']
    #day_night = data['day_night']
    is_holiday = data['is_holiday']

    # モデルに入力するためのデータを準備
    # ここでは、地図の範囲内でグリッド状に緯度経度を生成します
    # 例として、東京近辺の緯度経度を使用します

     # 緯度・経度の範囲を日本全体に設定
    lat_min = 24.0  # 日本の南端
    lat_max = 46.0  # 日本の北端
    lon_min = 123.0  # 日本の西端
    lon_max = 146.0  # 日本の東端

    # ユーザーの指定した時間を現在の日付と組み合わせて datetime 型に変換
    date_today = datetime.date.today()
    date_time = datetime.datetime.combine(date_today, datetime.time(hour=hour))
    date_time = jst.localize(date_time)

    # グリッドの間隔を設定（必要に応じて調整）
    num_points = 100  # 緯度・経度方向のポイント数
    lat_grid = np.linspace(lat_min, lat_max, num=num_points)
    lon_grid = np.linspace(lon_min, lon_max, num=num_points)

    grid_points = []
    for lat in lat_grid:
        for lon in lon_grid:
            grid_points.append((lat, lon))

    # 入力データフレームを作成
    input_data = pd.DataFrame(grid_points, columns=['latitude', 'longitude'])
    input_data['hour'] = hour
    input_data['weekday'] = datetime.datetime.now().weekday()
    input_data['is_holiday'] = is_holiday
    input_data['昼夜区分'] = day_night
    input_data['天候区分'] = weather

     # 緯度・経度から昼夜区分を計算
    day_night_list = []
    for idx, row in input_data.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        day_night = get_day_night_code(lat, lon, date_time)
        day_night_list.append(day_night)
    input_data['昼夜区分'] = day_night_list

    # カテゴリ変数のエンコード
    for column in ['昼夜区分', '天候区分']:
        le = label_encoders[column]
        input_data[column] = le.transform(input_data[column])

    # 'is_holiday'を数値に変換
    input_data['is_holiday'] = input_data['is_holiday'].astype(int)

    # フィーチャーの順序を訓練時と一致させる
    feature_columns = ['latitude', 'longitude', 'hour', 'weekday', 'is_holiday', '昼夜区分', '天候区分']
    input_data = input_data[feature_columns]

    # 事故リスクの予測
    risk_scores = model.predict_proba(input_data)[:, 1]
    input_data['risk_score'] = risk_scores

    # リスクスコアが高い地点をフィルタリング（任意）
    high_risk_data = input_data[input_data['risk_score'] > 0.5]

    # 必要な情報を辞書形式で返す
    risk_data = high_risk_data[['latitude', 'longitude', 'risk_score']].to_dict(orient='records')

    return jsonify({'riskData': risk_data})

if __name__ == '__main__':
    app.run(debug=True)
