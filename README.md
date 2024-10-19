Dự án Dự đoán Giá Cổ phiếu sử dụng LSTM

Dự án này sử dụng mạng neural LSTM (Long Short-Term Memory) để dự đoán giá cổ phiếu trong tương lai dựa trên dữ liệu lịch sử.
Tổng quan
Dự án này bao gồm các bước sau:

Thu thập dữ liệu cổ phiếu từ Yahoo Finance
Tiền xử lý và chuẩn hóa dữ liệu
Xây dựng và huấn luyện mô hình LSTM
Dự đoán giá cổ phiếu và đánh giá kết quả

Yêu cầu

Python 3.x
pandas
numpy
matplotlib
scikit-learn
tensorflow
yfinance
mplfinance
seaborn

Bạn có thể cài đặt các thư viện cần thiết bằng cách chạy:
Copypip install pandas numpy matplotlib scikit-learn tensorflow yfinance mplfinance seaborn
Cách sử dụng

Clone repository này về máy local của bạn.
Mở file notebook stock_price_prediction.ipynb trong Jupyter Notebook hoặc JupyterLab.
Chạy các cell trong notebook theo thứ tự để:

Thu thập dữ liệu cổ phiếu
Hiển thị biểu đồ nến và biểu đồ đường
Xây dựng và huấn luyện mô hình LSTM
Dự đoán giá cổ phiếu và hiển thị kết quả


Để thay đổi cổ phiếu được phân tích, hãy sửa đổi mã cổ phiếu trong dòng df = yf.Ticker("VIC.VN").
Bạn có thể điều chỉnh các tham số của mô hình như prediction_days và future để thay đổi khoảng thời gian dự đoán. ( sử dụng giao diện xây dựng bàng thư viện Streamlit)

Lưu ý
Dự án này chỉ nhằm mục đích nghiên cứu và giáo dục. Không nên sử dụng nó làm cơ sở duy nhất cho các quyết định đầu tư thực tế.
