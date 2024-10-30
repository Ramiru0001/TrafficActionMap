#get_amedas_weather.py
from datetime import datetime, timezone, timedelta
import requests
import math
def format_latest_time(latest_time_str):
    # ISO8601形式の日時文字列をパース
    dt = datetime.strptime(latest_time_str, '%Y-%m-%dT%H:%M:%S%z')
    # 希望の形式に変換
    time = dt.strftime('%Y%m%d%H%M%S')
    return time

def format_YYYYMMDD_h3(latest_time_str):
    dt = datetime.strptime(latest_time_str, '%Y-%m-%dT%H:%M:%S%z')
    # 最新の3時間ごとの時刻に丸める
    hour = (dt.hour // 3) * 3
    dt_rounded = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
    # 希望の形式でファイル名を生成
    time = dt_rounded.strftime('%Y%m%d_%H')
    return time

def format_YYYYMMDDHHMM00(latest_time_str):
    # ISO8601形式の日時文字列をパース
    dt = datetime.strptime(latest_time_str, '%Y-%m-%dT%H:%M:%S%z')
    # 希望の形式に変換
    time = dt.strftime('%Y%m%d%H%M00')
    return time

def get_nearest_station(lat, lon):
    # アメダス観測所の一覧を取得
    url = 'https://www.jma.go.jp/bosai/amedas/const/amedastable.json'
    response = requests.get(url)
    response.raise_for_status()
    stations = response.json()

    # 最も近い観測所を探す
    min_distance = float('inf')
    nearest_station_code = None
    for station_code, info in stations.items():
        station_lat = info['lat'][0]  # リストの最初の要素を取得
        station_lon = info['lon'][0]  # リストの最初の要素を取得
        # 距離を計算（簡易的に緯度経度の差を使用）
        distance = math.sqrt((lat - station_lat) ** 2 + (lon - station_lon) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_station_code = station_code
            
    print(f"nearest_station_code:{nearest_station_code}")
    return nearest_station_code

def get_current_weather(lat, lon):
    station_code = get_nearest_station(lat, lon)
    if not station_code:
        print("最寄りの観測所が見つかりませんでした。")
        return None

    # 最新の観測データの時刻を取得
    url = f'https://www.jma.go.jp/bosai/amedas/data/latest_time.txt'
    response = requests.get(url)
    response.raise_for_status()
    #latest_time = response.text.strip().replace('"', '')
    latest_time_str = response.text.strip()
    print(f"取得した最新時刻: {latest_time_str}")
    #print(f"latest_time:{latest_time}")

    # 日時文字列を希望の形式に変換
    latest_time = format_latest_time(latest_time_str)
    print(f"取得した時刻: {latest_time}")

    #形式変更
    latest_time_h3 = format_YYYYMMDD_h3(latest_time_str)
    print(f"取得した時刻h3: {latest_time_h3}")

    # 観測データを取得
    data_url = f'https://www.jma.go.jp/bosai/amedas/data/point/{station_code}/{latest_time_h3}.json'
    response = requests.get(data_url)
    response.raise_for_status()
    data = response.json()
    if data == None:
        return None

    #形式変更
    latest_time_YmdHM00 = format_YYYYMMDDHHMM00(latest_time_str)
    print(f"取得した最終記録時刻: {latest_time_YmdHM00}")

    return data[latest_time_YmdHM00]

    # データ内のキーが観測所コードであることを確認
    station_data = data.get(station_code)
    if not station_data:
        print("観測データが見つかりませんでした。")
        return None
    
    print("clear")
    # 必要な観測データを取得（例：気温、降水量、風速など）
    # temperature = data[latest_time]['temp'][0] if 'temp' in data[latest_time] else None
    # precipitation = data[latest_time]['precipitation1h'][0] if 'precipitation1h' in data[latest_time] else None
    # wind_speed = data[latest_time]['wind'][0] if 'wind' in data[latest_time] else None

    # current_weather = {
    #     'temperature': temperature,
    #     'precipitation': precipitation,
    #     'wind_speed': wind_speed
    # }

    # return current_weather
    return 0

if __name__ == '__main__':
    # テスト用の緯度・経度（例：東京駅）
    lat = 35.30642033308637
    lon = 133.98634905167242

    current_weather = get_current_weather(lat, lon)
    print(f"現在の天気観測データ: {current_weather}")
