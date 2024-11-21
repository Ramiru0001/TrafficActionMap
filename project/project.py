# project.py
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
import get_amedas_weather
import traceback
import math
import pytz
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import Point
import signal

app = Flask(__name__)
app.logger.debug('DEBUG')
# 環境変数から API キーを取得
api_key = os.environ.get('OPENWEATHERMAP_API_KEY')

# モデルとラベルエンコーダーの読み込み
model = joblib.load('accident_risk_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# クラスターの読み込み
cluster_polygons_gdf = gpd.read_file('cluster_polygons.geojson')
cluster_polygons_gdf = cluster_polygons_gdf.to_crs(epsg=4326)  # WGS84 座標系に変換

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

def get_weather_OpenWeatherMap(lat, lon,api_key):
    try:
        # get_OpenWeatherMap.py の get_current_weather 関数を呼び出す
        weather_info = get_OpenWeatherMap.get_current_weather(lat, lon, api_key)
        if weather_info:
            #print(f"取得した天気情報: {weather_info}")
            return weather_info
        else:
            print("OpenWeatherMap天気情報の取得に失敗しました。")
            return None
    except Exception as e:
        traceback.print_exc()  # エラーの詳細を表示
        return jsonify({'error': 'サーバーエラーが発生しました', 'details': str(e)}), 500
def get_weather_amedas(lat,lon):
    amedas_weather_info=get_amedas_weather.get_current_weather(lat,lon)
    if amedas_weather_info:
            #print(f"取得した天気情報: {weather_info}")
            return amedas_weather_info
    else:
        print("amedas天気情報の取得に失敗しました。")
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

@app.route('/get_weather', methods=['POST'])
def get_weather():
    try:
        print("get_weatherが呼ばれた")
        data = request.get_json()
        lat = data['lat']
        lon = data['lon']
        weather_data = get_weather_OpenWeatherMap(lat,lon,api_key)
        amedas_data = get_weather_amedas(lat,lon)
        weather=weather_data['weather_type'] 
        #print(f"緯度 {lat}、経度 {lon} の天気は: {weather} です。",flush=True)
        if not weather:
            return jsonify({'error': '天気データの取得に失敗しました'}), 500
        #1アメダスで、10分、もしくは1時間以内に雨が少しでも降っていたら雨
        if (amedas_data["precipitation1h"] or amedas_data["precipitation10m"]) and (amedas_data["precipitation1h"][0]>0 or amedas_data["precipitation10m"][0]>0):
            weather="雨"
        elif weather_data['weather_id'] // 100 == 6:
            weather = '雪'

        elif weather_data['visibility'] <=200:
            weather = '霧'

        elif weather_data['cloudiness']>=80:
            weather = '曇'
        else:
            weather = '晴'
        
        print(weather)
        return jsonify({'weather': weather})
    except Exception as e:
        traceback.print_exc()  # スタックトレースをコンソールに出力
        return jsonify({'error': 'サーバーエラーが発生しました', 'details': str(e)}), 500

def map_day_night(code):
    day_night_dict = {
        0: '不明', 11: '昼_明', 12: '昼_昼', 13: '昼_暮',
        21: '夜_暮', 22: '夜_夜', 23: '夜_明'
    }
    return day_night_dict.get(int(code), '不明')

# 天候と昼夜の区分をマッピング
def map_weather(code):
    weather_dict = {
        0: '不明', 1: '晴', 2: '曇', 3: '雨', 4: '霧', 5: '雪'
    }
    return weather_dict.get(int(code), '不明')
    
@app.route('/get_risk_data', methods=['POST'])
def get_risk_data():
    try:
        print(f"get_risk_dataが呼ばれました")
        data = request.get_json()
        weather_str = data['weather']
        datetime_str = data['datetime']
        latitude = data['latitude']
        longitude = data['longitude']
        prediction_radius=data['prediction_radius']#予想半径(m)

        # datetime_str を datetime オブジェクトに変換
        date_time = datetime.datetime.fromisoformat(datetime_str)
        date_time = jst.localize(date_time)  # 日本時間に設定
        
        # ユーザーの位置から指定された半径内のクラスターを取得
        user_location = Point(longitude, latitude)

        # ユーザーの位置をバッファリングして範囲を作成
        user_location_gdf = gpd.GeoDataFrame(geometry=[user_location], crs='EPSG:4326')
        user_location_buffer = user_location_gdf.to_crs(epsg=3857).buffer(prediction_radius).to_crs(epsg=4326).union_all()
        
        print(f"クラスターの取得")
        # 範囲内のクラスターを取得
        clusters_in_area = cluster_polygons_gdf[cluster_polygons_gdf.intersects(user_location_buffer)]

        if clusters_in_area.empty:
            return jsonify({'riskData': [], 'message': '指定された範囲内にクラスターが存在しません。'})
        
        print(f"入力データを作成")
        # 入力データを作成
        input_data_list = []

        for idx, cluster_row in clusters_in_area.iterrows():
            cluster_centroid = cluster_row['geometry'].centroid
            cluster_lat = cluster_centroid.y
            cluster_lon = cluster_centroid.x
            road_width = cluster_row['road_width']
            road_alignment = cluster_row['road_alignment']
            road_shape = cluster_row['road_shape']

            # 特徴量を計算
            is_holiday = int(jpholiday.is_holiday(date_time.date()))
            month = date_time.month
            day = date_time.day
            hour = date_time.hour
            minute = date_time.minute
            weekday = date_time.weekday()
            day_night_code = get_day_night_code(cluster_lat, cluster_lon, date_time)
            day_night = map_day_night(day_night_code)
            weather = weather_str

            # 特徴量の辞書を作成
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
                '天候区分': weather,  # 天候の文字列を使用
                'road_width': road_width,
                'road_alignment': road_alignment,
                'road_shape': road_shape,
                'cluster': cluster_row['cluster']
            }
            input_data_list.append(input_data)
        
        print(f"入力データフレームを作成")
        # 入力データフレームを作成
        input_df = pd.DataFrame(input_data_list)
        
        # カテゴリ変数のエンコード
        for column in ['昼夜区分', '天候区分', 'road_width', 'road_alignment', 'road_shape']:
            le = label_encoders[column]
            input_df[column] = le.transform(input_df[column])

        # 'cluster' 列を別途保存
        cluster_column = input_df['cluster']

        # フィーチャーの順序を訓練時と一致させる
        feature_columns = ['latitude', 'longitude', 'month', 'day', 'hour', 'weekday', '昼夜区分', '天候区分', 'is_holiday', 'road_width', 'road_alignment', 'road_shape']
        input_df = input_df[feature_columns]

        print(f"事故リスクの予測")
        # 事故リスクの予測
        risk_scores = model.predict_proba(input_df)[:, 1]

        # 元のデータにリスクスコアとクラスター列を戻す
        input_df['accident'] = risk_scores
        input_df['cluster'] = cluster_column

        print(f"結果を集計（同じクラスターについて平均を取る）")
        # 結果を集計（同じクラスターについて平均を取る）
        risk_summary = input_df.groupby(['cluster']).agg({
            'latitude': 'first',
            'longitude': 'first',
            'accident': 'mean'
        }).reset_index()

        # リスクスコアと位置情報を取得
        risk_data = risk_summary.to_dict(orient='records')
        #print(f"リスクデータ：{risk_data}")

        # 結果をクライアントに返す
        return jsonify({
            'riskData': risk_data
        })
    except Exception as e:
        traceback.print_exc()  # エラーの詳細を表示
        return jsonify({'error': 'サーバーエラーが発生しました', 'details': str(e)}), 500

def some_risk_prediction_function(lat, lon, weather, datetime):
    # 実際のリスク予測ロジックを実装
    # ダミーのリスクスコア
    return 0.5

if __name__ == '__main__':
    try:
        # Signal handler for graceful shutdown
        def handle_exit(signum, frame):
            print("\nGracefully shutting down the server...")
            exit(0)
        
        # Attach the signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, handle_exit)

        app.run(debug=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user (Ctrl+C).")
