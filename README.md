# Indoor Location System

Hệ thống định vị trong nhà sử dụng dữ liệu RSSI từ các beacon và vector similarity search.

## Yêu cầu hệ thống

- Python 3.10+
- Docker
- PostgreSQL (nếu sử dụng phiên bản PostgreSQL)
- Qdrant (nếu sử dụng phiên bản Qdrant)

## Cài đặt

1. Clone repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

2. Tạo và kích hoạt môi trường conda:
```bash
conda create -n indoor_location python=3.8
conda activate indoor_location
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu trúc dữ liệu

Dữ liệu đầu vào cần có định dạng CSV với các cột:
- Các cột RSSI của beacon (ví dụ: E2:19:18:C3:B9:B8, C6:21:3B:AE:36:45, ...)
- Cột label chứa vị trí tương ứng

## Các bước chạy hệ thống

### 1. Chuẩn bị dữ liệu

Chạy script để tách dữ liệu thành tập train và test:
```bash
python prepare_data_from_csv.py 
```

Script này sẽ:
- Đọc dữ liệu từ file CSV
- Tách dữ liệu thành tập train và test
- Lưu kết quả vào file JSON (processed_data.json và test_samples.json)

### 2. Chọn phiên bản hệ thống

Có hai phiên bản hệ thống:

#### Phiên bản PostgreSQL (sử dụng pgvector)

1. Khởi động PostgreSQL container:
```bash
docker-compose -f docker-compose.postgres.yml up -d
```

2. Chạy hệ thống:
```bash
python indoor_location_system_pgvector.py
```

#### Phiên bản Qdrant

1. Khởi động Qdrant container:
```bash
docker-compose -f docker-compose.qdrant.yml up -d
```

2. Chạy hệ thống:
```bash
python indoor_location_system.py
```

### 3. Đánh giá kết quả

Hệ thống sẽ:
- Tạo collection/table trong database
- Upload dữ liệu đã xử lý
- Đánh giá trên tập test
- In ra kết quả:
  - Độ chính xác tổng thể
  - Số lượng dự đoán
  - Số lượng dự đoán đúng
  - Chi tiết từng mẫu test

## Các tham số có thể điều chỉnh

Trong file `indoor_location_system.py` hoặc `indoor_location_system_pgvector.py`:

- `k`: Số lượng neighbors cho KNN (mặc định: 5)
- `distance_threshold`: Ngưỡng khoảng cách tối đa (mặc định: 10.0)
- `collection_name`: Tên collection trong Qdrant (mặc định: "indoor_locations")
- Các tham số kết nối database (host, port, username, password)

## Lưu ý

1. Đảm bảo Docker đang chạy trước khi khởi động container
2. Kiểm tra port không bị conflict (5432 cho PostgreSQL, 6333 cho Qdrant)
3. Dữ liệu đầu vào cần được chuẩn hóa để có kết quả tốt
4. Có thể điều chỉnh các tham số để tối ưu kết quả

## Xử lý lỗi thường gặp

1. Lỗi kết nối database:
   - Kiểm tra Docker container đang chạy
   - Kiểm tra thông tin kết nối (host, port, credentials)
   - Kiểm tra port không bị block

2. Lỗi dữ liệu:
   - Kiểm tra định dạng file CSV
   - Kiểm tra tên cột RSSI khớp với beacon_order
   - Kiểm tra dữ liệu không bị null/missing

3. Lỗi memory:
   - Giảm số lượng mẫu test
   - Tăng thời gian chờ giữa các request
   - Tối ưu hóa câu query 