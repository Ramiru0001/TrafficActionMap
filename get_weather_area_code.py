import requests
import pandas as pd

# エクセルファイルのパス
excel_file = 'data/000925835.xlsx'

# シート名を指定して読み込む（省略可）
df1 = pd.read_excel(excel_file, sheet_name='R6.1.1現在の団体')
df2 = pd.read_excel(excel_file, sheet_name='R6.1.1政令指定都市')
# 2つのデータを結合
df = pd.concat([df1, df2], ignore_index=True)
# データの確認
#print(df.head())

# 必要なカラムが存在するか確認
required_columns = ['団体コード']  # 必要なカラム名に修正
if not all(column in df.columns for column in required_columns):
    raise ValueError("エクセルファイルに必要なカラムが存在しません。")

def get_municipality_code(lat, lon):
    #緯度経度から自治体コードへの変換は国土地理院の逆ジオコーディングAPI
    url = f'https://mreversegeocoder.gsi.go.jp/reverse-geocoder/LonLatToAddress?lat={lat}&lon={lon}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results')
        if results:
            muniCd = results.get('muniCd')
            return muniCd
        else:
            print("No results found for the given latitude and longitude.")
            return None
    else:
        print(f"API request failed with status code {response.status_code}")
        return None

def get_city_level_code(muniCd):
    # チェックディジットを削除して5桁のコードにする
    city_code = muniCd[:5]
    return city_code

#自治体コードを市レベルのコードに変換
def get_area_code(muniCd):
    # 市レベルの自治体コードに変換
    city_code = get_city_level_code(muniCd)

    # area.jsonを取得
    url = 'https://www.jma.go.jp/bosai/common/const/area.json'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to get area.json with status code {response.status_code}")
        return None
    area_data = response.json()

    # 'class20s'セクションから地域コードを取得
    class20s = area_data.get('class20s', {})
    for area_code, info in class20s.items():
        muniCd_in_info = info.get('muniCd')
        if muniCd_in_info:
            # muniCd_in_infoも5桁にして比較
            muniCd_in_info_city = muniCd_in_info[:5]
            if muniCd_in_info_city == city_code:
                return area_code  # 地域コードを返す
    #以下の文章がいつも出力されている
    print(f"Area code not found for city-level muniCd: {city_code}")
    return None

def get_weather_area_code(lat, lon):
    muniCd = get_municipality_code(lat, lon)
    if not muniCd:
        print("Failed to get municipality code.")
        return None

    area_code = get_area_code(muniCd)
    if not area_code:
        print("Failed to get area code.")
        return None

    return area_code

if __name__ == '__main__':
    # テスト用の緯度・経度（例：東京駅）
    lat = 35.681236
    lon = 139.767125

    area_code = get_weather_area_code(lat, lon)
    if area_code:
        print(f"The weather forecast area code for latitude {lat} and longitude {lon} is: {area_code}")
    else:
        print("Could not retrieve the area code.")
