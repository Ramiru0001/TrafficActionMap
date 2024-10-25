#get_OpenWeatherMap.py
import requests

def get_current_weather(lat, lon, api_key):
    # OpenWeatherMapのCurrent Weather Data APIエンドポイント
    url = 'https://api.openweathermap.org/data/2.5/weather'

    # パラメータの設定
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric',      # 摂氏温度を取得
        'lang': 'ja'            # 日本語の天気説明を取得
    }

    try:
        # APIへのリクエスト
        response = requests.get(url, params=params)
        response.raise_for_status()  # ステータスコードが200番台でない場合、例外を発生させる

        # JSONデータの解析
        data = response.json()

        # 必要な情報の抽出
        weather_description = data['weather'][0]['description']  # 天気の説明
        weather_id = data['weather'][0]['id']                    # 天気のID
        temperature = data['main']['temp']                      # 気温
        humidity = data['main']['humidity']                     # 湿度
        wind_speed = data['wind']['speed']                      # 風速
        cloudiness = data['clouds']['all']                      # 雲量（パーセント）
        visibility = data.get('visibility', None)                # 視程（メートル）
        # 天気IDが雷雨、霧雨、雨、雪の場合の判定
        if weather_id // 100 == 2:
            weather_type = '雨'

        elif weather_id // 100 == 3:
            weather_type = '雨'

        elif weather_id // 100 == 5:
            weather_type = '雨'

        elif weather_id // 100 == 6:
            weather_type = '雪'

        elif visibility <=200:
            weather_type = '霧'

        elif cloudiness>=80:
            weather_type = '曇'

        else:
            weather_type = '晴'

        # 結果の辞書を作成
        current_weather = {
            'weather': weather_description,
            'weather_id': weather_id,
            'weather_type': weather_type,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'cloudiness': cloudiness,
            'visibility': visibility
        }

        return current_weather

    except requests.exceptions.HTTPError as errh:
        print(f"HTTPエラー: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"接続エラー: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"タイムアウトエラー: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"リクエストエラー: {err}")
    except Exception as e:
        print(f"予期せぬエラー: {e}")

    return None

if __name__ == '__main__':
    # 例として東京駅の緯度・経度を使用
    lat = 34.329506911402255
    lon = 134.0367265314154

    # OpenWeatherMapのAPIキーをここに入力
    api_key = 'caff0feac5f8764f6a1f474405a4f99e'

    # 現在の天気情報を取得
    weather_info = get_current_weather(lat, lon, api_key)

    if weather_info:
        print("現在の天気情報:")
        print(f"天気: {weather_info['weather']}")
        print(f"天気ID: {weather_info['weather_id']}")
        print(f"天気タイプ: {weather_info['weather_type']}")
        print(f"気温: {weather_info['temperature']}°C")
        print(f"湿度: {weather_info['humidity']}%")
        print(f"風速: {weather_info['wind_speed']} m/s")
        print(f"雲量: {weather_info['cloudiness']}%")
        print(f"視程: {weather_info['visibility']} m")
    else:
        print("天気情報の取得に失敗しました。")
