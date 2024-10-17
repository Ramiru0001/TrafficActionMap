#get_weather_area_code.py
import requests
import pandas as pd
import sys
import io

# 標準出力のエンコーディングをUTF-8に設定
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# エクセルファイルのパス
excel_file = 'data/000925835.xlsx'

# シート名を指定して読み込む（省略可）
df = pd.read_excel(excel_file, sheet_name='R6.1.1現在の団体')
# データの確認
#print(df.head())

# 必要なカラムが存在するか確認
required_columns = ['団体コード']  # 必要なカラム名に修正
if not all(column in df.columns for column in required_columns):
    raise ValueError("エクセルファイルに必要なカラムが存在しません。")

# '団体コード' のリストを取得し、6桁目を削除して5桁のコードに変換
# リーディングゼロを削除して整数に変換
city_codes = df['団体コード'].astype(str).apply(lambda x: int(x[:-1].lstrip('0'))).tolist()
city_codes.sort()

# city_codesの最小値と最大値を出力
print(f"city_codesの最小値: {min(city_codes)}, 最大値: {max(city_codes)}")

def get_municipality_code(lat, lon):
    #緯度経度から自治体コードへの変換は国土地理院の逆ジオコーディングAPI
    url = f'https://mreversegeocoder.gsi.go.jp/reverse-geocoder/LonLatToAddress?lat={lat}&lon={lon}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        results = data.get('results')
        if results:
            muniCd = results.get('muniCd')
            print(f"取得した自治体コード (muniCd): {muniCd}")  # muniCdを出力
            return muniCd
        else:
            print("指定した緯度・経度に対応する自治体コードが見つかりませんでした。")
            return None
    else:
        print(f"APIリクエストが失敗しました。ステータスコード: {response.status_code}")
        return None

def local_code_to_city_code(code):
    # コードの6桁目を削除して5桁のコードにする
    code_adj = code[:5]
    # リーディングゼロを削除して整数に変換
    code_int = int(code_adj.lstrip('0'))
    print(f"調整後の自治体コード (整数値): {code_int}")

    # city_codes内でcode_int付近の値を出力（デバッグ用）
    index = city_codes.index(code_int) if code_int in city_codes else -1
    if index != -1:
        nearby_codes = city_codes[max(0, index-5):min(len(city_codes), index+5)]
        print(f"city_codes内の周辺コード: {nearby_codes}")
    else:
        print(f"code_int ({code_int}) は city_codes に存在しません。")
    
    # 二分探索で同じ値か、それより大きい最小の値を探す
    left, right = 0, len(city_codes) - 1
    nearest_code_int = None
    while left <= right:
        mid = (left + right) // 2
        mid_code_int = city_codes[mid]
        print(f"left: {left}, right: {right}, mid: {mid}, mid_code_int: {mid_code_int}")
        if mid_code_int == code_int:
            nearest_code_int = mid_code_int
            print(f"一致するコードが見つかりました: {nearest_code_int}")
            break
        elif mid_code_int < code_int:
            left = mid + 1
        else:
            nearest_code_int = mid_code_int
            right = mid - 1
    if nearest_code_int is None:
        # 一致するコードやそれより大きいコードが見つからなかった場合
        print(f"一致するコードやそれより大きいコードが見つかりませんでした。muniCd: {code_adj}")
        return None

    # 整数を5桁の文字列に戻す（必要に応じて先頭にゼロを付加）
    adjusted_code = str(nearest_code_int).zfill(5)
    print(f"最も近いコード: {adjusted_code}")
    return adjusted_code
#自治体コード(区レベルコード)を市レベル（少し荒い）のコードに変換
def get_area_code(muniCd):
    # 市レベルの自治体コードに変換
    city_code = local_code_to_city_code(muniCd)
    if not city_code:
        print("市レベルの自治体コードへの変換に失敗しました。")
        return None
    
    # area.jsonを取得
    url = 'https://www.jma.go.jp/bosai/common/const/area.json'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"area.json の取得に失敗しました。ステータスコード: {response.status_code}")
        return None
    area_data = response.json()

    # 'class20s'セクションから地域コードを取得
    class20s = area_data.get('class20s', {})
    # city_codeの末尾に'00'から'09'を付加して探索
    for suffix in ['{:02d}'.format(i) for i in range(10)]:
        candidate_code = city_code + suffix
        print(f"Trying candidate_code: {candidate_code}")
        if candidate_code in class20s:
            area_code = candidate_code
            return area_code  # 地域コードを返す
    print(f"地域コードが見つかりませんでした。市レベルのコード: {city_code}")
    return None

def get_weather_area_code(lat, lon):
    muniCd = get_municipality_code(lat, lon)
    if not muniCd:
        print("自治体コードの取得に失敗しました。")
        return None

    area_code = get_area_code(muniCd)
    if not area_code:
        print("地域コードの取得に失敗しました。")
        return None

    return area_code

if __name__ == '__main__':
    # テスト用の緯度・経度（例：東京駅）
    lat = 34.354710503107086
    lon = 136.35887471608626

    area_code = get_weather_area_code(lat, lon)
    if area_code:
        print(f"緯度 {lat}、経度 {lon} の天気予報地域コードは: {area_code} です。")
    else:
        print("地域コードを取得できませんでした。")
