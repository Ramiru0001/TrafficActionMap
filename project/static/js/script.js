// static/js/script.js

// 地図の初期化
var map = L.map('map').setView([35.0, 135.0], 5); // デフォルトの位置

// タイルレイヤーの追加
// L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
//     attribution: '© OpenStreetMap contributors'
// }).addTo(map);

// マーカーを保持する変数
var marker;

// 地図とマーカーを初期化
initializeMap();

// 地図の初期化と現在位置の取得を行う関数
function initializeMap() {
    // デフォルトの位置（東京駅の座標）
    var defaultLat = 35.681236;
    var defaultLon = 139.767125;

    // Geolocation APIを使用して現在位置を取得
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            // 位置情報の取得に成功した場合
            var lat = position.coords.latitude;
            var lon = position.coords.longitude;

            // 地図を現在位置に設定
            map.setView([lat, lon], 15);

            // マーカーを追加
            addMarker(lat, lon);

            // 住所を表示
            displayAddress(lat, lon);

            // 選択した位置の天気と危険情報を取得
            getWeatherAndRiskData(lat, lon);

        }, function(error) {
            // 位置情報の取得に失敗した場合
            console.error('Error Code: ' + error.code + ' - ' + error.message);

            // デフォルト位置に地図を設定
            map.setView([defaultLat, defaultLon], 12);

            // マーカーを追加
            addMarker(defaultLat, defaultLon);

            // 住所を表示
            displayAddress(defaultLat, defaultLon);

            // 選択した位置の天気と危険情報を取得
            getWeatherAndRiskData(defaultLat, defaultLon);
        });
    } else {
        // Geolocation APIが使用できない場合
        alert('このブラウザでは位置情報がサポートされていません。');

        // デフォルト位置に地図を設定
        map.setView([defaultLat, defaultLon], 12);

        // マーカーを追加
        addMarker(defaultLat, defaultLon);

        // 住所を表示
        displayAddress(defaultLat, defaultLon);

        // 選択した位置の天気と危険情報を取得
        getWeatherAndRiskData(defaultLat, defaultLon);
    }

    // タイルレイヤーの追加（地図を初期化した後に追加）
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(map);
}

// マーカーを追加する関数
function addMarker(lat, lon) {
    // 既存のマーカーを削除
    if (marker) {
        map.removeLayer(marker);
    }

    // 新しいマーカーを追加
    marker = L.marker([lat, lon], { draggable: true }).addTo(map);

    // マーカーのドラッグイベントを追加
    addMarkerDragEvent(marker);
}

// マーカーをドラッグしたときのイベントリスナーを追加する関数
function addMarkerDragEvent(marker) {
    marker.on('dragend', function(e) {
        var lat = e.target.getLatLng().lat;
        var lon = e.target.getLatLng().lng;

        // 住所を更新
        displayAddress(lat, lon);

        // 選択した位置の天気と危険情報を取得
        getWeatherAndRiskData(lat, lon);
    });
}

// 地図をクリックしたときのイベントリスナー
map.on('click', function(e) {
    var lat = e.latlng.lat;
    var lon = e.latlng.lng;
    // マーカーを追加
    addMarker(lat, lon);

    // 画面に表示される住所を更新
    displayAddress(lat, lon);

    // 選択した位置の天気と危険情報を取得
    getWeatherAndRiskData(lat, lon);
});

// 画面上に住所を表示する関数
function displayAddress(lat, lon) {
    // Nominatim APIを使用して逆ジオコーディング
    var url = 'https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=' + lat + '&lon=' + lon + '&accept-language=ja';

    fetch(url)
    .then(response => response.json())
    .then(data => {
        if (data && data.address) {
            var address = data.address;
            var displayName = data.display_name;

            // display_nameをカンマで分割して配列にする
            var addressComponents = displayName.split(',');

            // 前後の空白を削除
            addressComponents = addressComponents.map(function(component) {
                return component.trim();
            });

            // 最後の要素が国名の場合、それを除外
            if (addressComponents[addressComponents.length - 1] === '日本') {
                addressComponents.pop();
            }

            // 次に最後の要素が郵便番号の場合、それを除外
            var postalCodePattern = /^\d{3}-\d{4}$/; // 郵便番号の正規表現パターン
            if (postalCodePattern.test(addressComponents[addressComponents.length - 1])) {
                addressComponents.pop();
            }

            // 残った要素を逆順（下から順）にする
            addressComponents = addressComponents.reverse();

            // 各要素からピリオドや不要な記号を除去（必要に応じて）
            addressComponents = addressComponents.map(function(component) {
                return component.replace('.', '',',');
            });

            // 住所要素の取得
            var postcode = address.postcode || '';

            // 住所の組み立て
            var formattedAddress = '';
            
            if (postcode) formattedAddress += '〒' + postcode + ' ';
            formattedAddress += addressComponents.join(' ');

            document.getElementById('addressDisplay').innerText = '現在の住所: ' + formattedAddress;
        } else {
            document.getElementById('addressDisplay').innerText = '住所を取得できませんでした';
        }

    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('addressDisplay').innerText = '住所の取得に失敗しました';
    });
}

// ディスプレイ上に引数と同じ天気情報を表示する関数
function displayWeather(weatherInfo) {
    //alert("天気更新")
    var weatherSelect = document.getElementById('weather');
    // 天気情報から選択肢を作成
    // weatherInfo がオブジェクトの場合、その中の 'weather_type' や 'weather' を使用します
    // ここでは、weatherInfo が文字列または配列である場合を考慮します
    // 天気情報が文字列の場合
    if (typeof weatherInfo === 'string') {
        // 選択肢をループして一致するものを選択
        for (var i = 0; i < weatherSelect.options.length; i++) {
            if (weatherSelect.options[i].value === weatherInfo) {
                weatherSelect.selectedIndex = i;
                break;
            }
        }
        console.log("weatherInfo=string");
    }
    // 天気情報が取得できない場合の処理（必要に応じて）
    else {
        var option = document.createElement('option');
        option.value = '';
        option.text = '天気情報を取得できませんでした';
        //weatherSelect.appendChild(option);
        console.log("weatherInfo=")
    }
}

// 地図の初期化
//var map = L.map('map').setView([35.681236, 139.767125], 12); // 東京駅を中心に設定

// OpenStreetMapのタイルレイヤーを追加
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors'
}).addTo(map);


// 選択した位置の天気と、危険情報を取得する関数
function getWeatherAndRiskData(lat, lon) {
    //alert('getWeatherAndRiskDataが呼ばれた');
    fetch('/get_weather', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ lat: lat, lon: lon })
    })
    .then(response => response.json())
    .then(data => {
        //alert(data.weather,data.riskData)
        // 天気情報の表示
        displayWeather(data.weather);

        // 地図上に危険情報を表示
        getRiskData();

    })
    .catch(error => {
        console.error('Error:', error);
        alert('天気データの取得に失敗しました。');
    });
}

//リスクデータを取得して表示
function getRiskData() {
    try{

        // 分析中メッセージを表示 編集
        const loadingMessage = document.getElementById('loadingMessage');
        loadingMessage.style.display = 'block';
        
        // フォームの値を取得
        var weatherSelect = document.getElementById('weather');
        var weather = weatherSelect.value;

        var dateInput = document.getElementById('date').value;
        //編集
        var timeInput = document.getElementById('time').value;
        timeInput = timeInput+":00" ;

        // 日付と時間を結合して datetime を作成
        var datetime = dateInput + 'T' + timeInput; // ISO形式の日時文字列
        //alert(datetime)
        // 半径予想範囲を取得
        var radiusSelect = document.getElementById('prediction_radius');
        var prediction_radius;
        if (radiusSelect.value === 'other') {
            var radiusInput = document.getElementById('prediction_radius_input');
            prediction_radius = parseInt(radiusInput.value);
            if (isNaN(prediction_radius)) {
                alert('半径予想範囲を正しく入力してください。');
                return;
            }
        } else {
            prediction_radius = parseInt(radiusSelect.value);
        }
        // 現在のマーカーの位置（緯度・経度）を取得
        var lat, lon;
        if (marker) {
            lat = marker.getLatLng().lat;
            lon = marker.getLatLng().lng;
        } else {
            alert('地図上にマーカーがありません。位置を選択してください。');
            return;
        }

        // サーバーにデータを送信
        var requestData = {
            weather: weather,
            datetime: datetime,
            latitude: lat,
            longitude: lon,
            prediction_radius:prediction_radius
        };
        fetch('/get_risk_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
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
            // データの内容を確認
            //alert('Received data:', data);
            if (data.riskData && data.riskData.length > 0) {
                // リスクデータを地図に表示
                displayRiskData(data.riskData);
            } else {
                console.log('指定された範囲内にリスクデータがありません。');
            }
            //alert(JSON.stringify(data.riskData));  // デバッグ用にデータを表示
            // リスクデータを地図に表示
            // リスクデータを地図に表示
            displayRiskData(data.riskData);
            //loadingMessage.style.display = 'none';
        })
        .catch(error => {
            console.error('Error:', error);
            //alert('リスクデータの取得に失敗しました。');
        });
    } catch(e) {
        alert( e.message );
    }
    loadingMessage.style.display = 'none';
}

var riskLayer;  // グローバル変数としてリスクレイヤーを保持

function displayRiskData(riskData) {
    try {
        // 既存のリスクレイヤーを削除
        if (riskLayer) {
            map.removeLayer(riskLayer);
        }
        // リスクデータをヒートマップ用に変換
        var heatData = riskData.map(function(dataPoint) {
            if (dataPoint.latitude && dataPoint.longitude && dataPoint.accident !== undefined) {
                return [dataPoint.latitude, dataPoint.longitude, dataPoint.accident];  // [緯度, 経度, リスクスコア]
            } else {
                console.warn('Invalid data point:', dataPoint);
                return null;
            }
        }).filter(Boolean);  // 無効なデータポイントを除外

        // ヒートマップレイヤーを作成して地図に追加
        riskLayer = L.heatLayer(heatData, {
            radius: 25,
            blur: 15,
            maxZoom: 17,
            max: 1,
            gradient: {
                0.0: 'green',   // リスクスコアが低い場合の色
                0.5: 'orange', // リスクスコアが中程度の場合の色
                1.0: 'red'     // リスクスコアが高い場合の色
            }
        }).addTo(map);
        console.log('Heatmap layer added.');
    } catch (error) {
        console.error('Error in displayRiskData:', error);
        alert('リスクデータの表示に失敗しました。');
    }
}

// 住所検索機能
function geocodeAddress() {
    var address = document.getElementById('addressInput').value;
    // ジオコーディングAPIを使用して住所を緯度経度に変換
    fetch('https://nominatim.openstreetmap.org/search?format=json&q=' + encodeURIComponent(address) + '&accept-language=ja')
    .then(response => response.json())
    .then(data => {
        if (data.length > 0) {
            var lat = parseFloat(data[0].lat);
            var lon = parseFloat(data[0].lon);
            map.setView([lat, lon], 14);

            // マーカーを追加
            addMarker(lat, lon);

            // 住所を更新
            displayAddress(lat, lon);

            // 選択した位置の天気と危険情報を取得
            getWeatherAndRiskData(lat, lon);
        } else {
            alert('住所が見つかりませんでした。');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('住所の検索に失敗しました。');
    });
}

// 半径予想範囲の表示切替関数
function toggleRadiusInput() {
    //console.log("a")
    var select = document.getElementById('prediction_radius');
    var inputDiv = document.getElementById('radius_input_div');
    //inputDiv.style.display = 'block';
    if (select.value === 'other') {
        inputDiv.style.display = 'block';
    } else {
        inputDiv.style.display = 'none';
    }
}

function resetall(){
    // 現在の日付と時間を取得
    //alert("resetall")
    var now = new Date();

    // 日付入力欄に今日の日付を設定
    var dateInput = document.getElementById('date').value;
    var year = now.getFullYear();
    var month = ('0' + (now.getMonth() + 1)).slice(-2);
    var day = ('0' + now.getDate()).slice(-2);
    dateInput = year + '-' + month + '-' + day;
    document.getElementById('date').value=dateInput;

    // 時間入力欄に現在の時間を設定
    var timeInput = document.getElementById('time');
    var hours = ('0' + now.getHours()).slice(-2);
    timeInput.value = hours;
    //マーカーから現在の位置情報を取得
    var lat, lon;
    if (marker) {
        lat = marker.getLatLng().lat;
        lon = marker.getLatLng().lng;
    } else {
        alert('地図上にマーカーがありません。位置を選択してください。');
        return;
    }
    // 選択した位置の天気と危険情報を取得
    getWeatherAndRiskData(lat,lon);

}

// var durationSelect = document.getElementById('prediction_duration');
//     var prediction_duration = parseInt(durationSelect.value);