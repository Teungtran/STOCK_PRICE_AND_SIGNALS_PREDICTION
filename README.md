## 📈 Dự án Dự đoán Giá Cổ phiếu (AAPL)
- LSTM (Tỷ lệ chính xác: 97%)
- Đánh Giá Tín hiệu sử dụng RandomForestClassifier (Mua, Bán, Giữ) (Tỷ lệ chính xác: 91%)

## 🔍 Tổng quan
Dự án này sử dụng mạng neural LSTM (Long Short-Term Memory) để dự đoán giá cổ phiếu trong tương lai dựa trên dữ liệu lịch sử, kết hợp với RandomForestClassifier để đưa ra tín hiệu giao dịch.

## 🌟 Tính năng chính
- Thu thập dữ liệu cổ phiếu tự động từ Yahoo Finance
- Tính toán các chỉ báo kỹ thuật:
  - RSI
  - SMA20, SMA50, SMA200
  - ATR
  - Bollinger Bands (Upper, Lower, Middle)
  - MACD
- Tiền xử lý và chuẩn hóa dữ liệu theo thời gian
- Dự đoán giá sử dụng mô hình LSTM
- Phân tích tín hiệu giao dịch bằng RandomForestClassifier
- Giao diện người dùng thân thiện với Streamlit

## 🛠 Yêu cầu hệ thống
### Thư viện Python cần thiết:
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- yfinance
- mplfinance
- seaborn
- streamlit

## 📦 Cài đặt
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow yfinance mplfinance seaborn streamlit
```

## 🚀 Hướng dẫn sử dụng
1. Thay đổi mã cổ phiếu phân tích tại mục "Stock symbol"
2. Điều chỉnh các tham số:
   - prediction_days: Số ngày dùng để dự đoán
   - future: Khoảng thời gian dự đoán trong tương lai
3. Tương tác với giao diện Streamlit để xem kết quả

## 📊 Hiệu suất
- Độ chính xác dự đoán giá (LSTM): 97%
- Độ chính xác tín hiệu giao dịch (RandomForestClassifier): 91%

## ⚠️ Hạn chế
- Chỉ có thể dự đoán chính xác các mã cổ phiếu cùng đơn vị tiền tệ
  - VD: Mô hình được huấn luyện trên dữ liệu USD chỉ phù hợp với các cổ phiếu giao dịch bằng USD
  - Không phù hợp cho các mã chứng khoán sử dụng đơn vị tiền tệ khác (như VND)

## 🔒 Tuyên bố miễn trừ trách nhiệm
Dự án này chỉ phục vụ mục đích nghiên cứu và giáo dục. Không nên sử dụng kết quả dự đoán làm cơ sở duy nhất cho các quyết định đầu tư thực tế.

## 🔧 Cấu hình khuyến nghị
```yaml
model_settings:
  lstm_units: 50
  dropout_rate: 0.2
  epochs: 10
  batch_size: 32
  
prediction_settings:
  prediction_days: 60
  future_prediction: 30
  
training_settings:
  train_test_split: 0.8
  validation_split: 0.1
```

## 📈 Ví dụ kết quả
- Dự đoán giá AAPL trong 30 ngày tới
- Tín hiệu giao dịch dựa trên phân tích kỹ thuật
- Biểu đồ so sánh giá dự đoán và giá thực tế

## 🙏 Lời cảm ơn
- Yahoo Finance API
- TensorFlow Team
- Cộng đồng Streamlit
- Các nhà đóng góp mã nguồn mở
