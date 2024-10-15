// static/js/script.js

// 地図の初期化
var map = L.map('map').setView([35.0, 135.0], 5); // デフォルトの位置

// タイルレイヤーの追加
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// マーカーを保持する変数
var marker;

// 地図をクリックしたときのイベントリスナー
map.on('click', function(e) {
    var lat = e.latlng.lat;
    var lon = e.latlng.lng;

    // 既存のマーカーを削除
    if (marker) {
        map.removeLayer(marker);
    }

    // 新しいマーカーを追加
    marker = L.marker([lat, lon], { draggable: true }).addTo(map);

    // マーカーのドラッグイベントを追加
    addMarkerDragEvent(marker);

    // 選択した位置の天気と危険情報を取得
    getWeatherAndRiskData(lat, lon);
});

fetch('/get_weather_and_risk_data', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ lat: lat, lon: lon })
})
.then(response => {
    if (!response.ok) {
        return response.json().then(errorData => {
            throw new Error(errorData.error || 'サーバーエラーが発生しました');
        });
    }
    return response.json();
})
.then(data => {
    // 天気情報の表示
    displayWeather(data.weather);

    // 地図上に危険情報を表示
    displayRiskDataOnMap(data.riskData);

    // 現在地を更新
    updateCurrentLocation(lat, lon);
})
.catch(error => {
    alert('エラー: ' + error.message);
});


// マーカーをドラッグしたときのイベントリスナーを追加する関数
function addMarkerDragEvent(marker) {
    marker.on('dragend', function(e) {
        var lat = e.target.getLatLng().lat;
        var lon = e.target.getLatLng().lng;

        // 選択した位置の天気と危険情報を取得
        getWeatherAndRiskData(lat, lon);
    });
}

function getWeatherCode(weatherText) {
    const weatherDict = {
        '晴れ': 1,
        '曇り': 2,
        '雨': 3,
        '霧': 4,
        '雪': 5
    };
    return weatherDict[weatherText] || 0;  // '0'は未知の天候
}

function getRiskPrediction(lat, lon, weatherText) {
    var date = new Date();
    var datetime = date.toISOString();  // ISO形式の日時文字列

    var weatherCode = getWeatherCode(weatherText);

    fetch('/predict_accident_risk', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            lat: lat,
            lon: lon,
            datetime: datetime,
            weather: weatherCode
        })
    })
    .then(response => response.json())
    .then(data => {
        displayRiskOnMap(lat, lon, data.riskScore);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


function getRiskPrediction(lat, lon, hour, weather, isHoliday, dayNight) {
    // 現在の曜日を取得
    var date = new Date();
    var weekday = date.getDay(); // 0（日曜日）から6（土曜日）

    fetch('/predict_accident_risk', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            lat: lat,
            lon: lon,
            hour: hour,
            weekday: weekday,
            is_holiday: isHoliday,
            day_night: dayNight,
            weather: weather
        })
    })
    .then(response => response.json())
    .then(data => {
        displayRiskOnMap(lat, lon, data.riskScore);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// マーカーをドラッグしたときのイベントリスナー
// map.on('moveend', function() {
//     if (marker) {
//         var lat = marker.getLatLng().lat;
//         var lon = marker.getLatLng().lng;

//         // 選択した位置の天気と危険情報を取得
//         getWeatherAndRiskData(lat, lon);
//     }
// });

// 天気情報を表示する関数
function displayWeather(weatherList) {
    var weatherText = weatherList.join('、'); // リストをカンマで結合
    document.getElementById('weatherInfo').innerText = '天気: ' + weatherText;
}


// 住所から位置を取得する関数
// function geocodeAddress() {
//     var address = document.getElementById('addressInput').value;
//     var geocoderUrl = 'https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(address);

//     fetch(geocoderUrl)
//         .then(response => response.json())
//         .then(data => {
//             if (data.length > 0) {
//                 var lat = parseFloat(data[0].lat);
//                 var lon = parseFloat(data[0].lon);

//                 // 地図の中心を移動
//                 map.setView([lat, lon], 13);

//                 // 既存のマーカーを削除
//                 if (marker) {
//                     map.removeLayer(marker);
//                 }

//                 // 新しいマーカーを追加
//                 marker = L.marker([lat, lon], { draggable: true }).addTo(map);

//                 // 選択した位置の天気と危険情報を取得
//                 getWeatherAndRiskData(lat, lon);
//             } else {
//                 alert('住所が見つかりませんでした。');
//             }
//         });
// }

// 地図の初期化
var map = L.map('map').setView([35.681236, 139.767125], 12); // 東京駅を中心に設定

// OpenStreetMapのタイルレイヤーを追加
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);

// 住所検索機能
function geocodeAddress() {
    var address = document.getElementById('addressInput').value;
    // ジオコーディングAPIを使用して住所を緯度経度に変換
    fetch('https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(address))
    .then(response => response.json())
    .then(data => {
        if (data.length > 0) {
            var lat = parseFloat(data[0].lat);
            var lon = parseFloat(data[0].lon);
            map.setView([lat, lon], 14);
        } else {
            alert('住所が見つかりませんでした。');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('住所の検索に失敗しました。');
    });
}


// 天気と危険情報を取得する関数
function getWeatherAndRiskData(lat, lon) {
    fetch('/get_weather_and_risk_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ lat: lat, lon: lon })
    })
    .then(response => response.json())
    .then(data => {
        // 天気情報の表示
        displayWeather(data.weather);

        // 地図上に危険情報を表示
        displayRiskDataOnMap(data.riskData);

        // 現在地を更新
        updateCurrentLocation(lat, lon);
    });
}

// 天気情報を表示する関数
function displayWeather(weather) {
    document.getElementById('weatherInfo').innerText = '天気: ' + weather;
}

// 地図上に危険情報を表示する関数
function displayRiskDataOnMap(riskData) {
    // リスクマーカーを追加（実装は省略）
}

// 現在地を更新する関数
function updateCurrentLocation(lat, lon) {
    // 必要に応じて現在地を保持
}

function getRiskData() {
    // フォームの値を取得
    var weather = document.getElementById('weather').value;
    var hour = parseInt(document.getElementById('hour').value);
    var is_holiday = parseInt(document.getElementById('is_holiday').value);

    // 現在の地図の表示範囲を取得
    var bounds = map.getBounds();
    var lat_min = bounds.getSouthWest().lat;
    var lat_max = bounds.getNorthEast().lat;
    var lon_min = bounds.getSouthWest().lng;
    var lon_max = bounds.getNorthEast().lng;

    // サーバーにデータを送信
    var requestData = {
        weather: weather,
        hour: hour,
        day_night: day_night,
        is_holiday: is_holiday,
        lat_min: lat_min,
        lat_max: lat_max,
        lon_min: lon_min,
        lon_max: lon_max
    };

    fetch('/get_risk_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        // リスクデータを地図に表示
        displayRiskData(data.riskData);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('リスクデータの取得に失敗しました。');
    });
}

var riskMarkers = []; // 既存のリスクマーカーを保持

function displayRiskData(riskData) {
    // 既存のリスクマーカーを削除
    riskMarkers.forEach(function(marker) {
        map.removeLayer(marker);
    });
    riskMarkers = [];

    // リスクデータをマップに表示
    riskData.forEach(function(point) {
        var lat = point.latitude;
        var lon = point.longitude;
        var risk = point.risk_score;

        var color = risk > 0.7 ? 'red' : 'orange';

        var marker = L.circleMarker([lat, lon], {
            radius: 5,
            color: color,
            fillOpacity: 0.7
        }).addTo(map);

        riskMarkers.push(marker);
    });
}


