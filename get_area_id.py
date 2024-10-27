#get_area_id.py
import sys
sys.path.append("C:\\Users\\shian\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python312\\site-packages")
import geopandas as gpd
from shapely.geometry import Point

# エリアポリゴンを読み込む
japan_grids = gpd.read_file('japan_area_grids.geojson')

def get_area_id(latitude, longitude):
    point = Point(longitude, latitude)
    matched_areas = japan_grids[japan_grids.contains(point)]
    if not matched_areas.empty:
        area_id = matched_areas.iloc[0]['area_id']
        return area_id
    else:
        return None  # 該当するエリアがない場合

# 使用例
latitude = 35.6895   # 緯度（例：東京駅付近）
longitude = 139.6917 # 経度

area_id = get_area_id(latitude, longitude)
if area_id is not None:
    print(f"この地点のエリアIDは: {area_id}")
else:
    print("指定された地点はエリア内に含まれていません。")
