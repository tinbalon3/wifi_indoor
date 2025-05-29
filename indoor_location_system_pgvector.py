import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import psycopg2
from psycopg2.extras import execute_values
from collections import Counter
import random

class IndoorLocationSystemPGVector:
    def __init__(self, dbname="indoor_location", user="postgres", password="postgres", host="localhost", port="5432"):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.beacon_order = [
            "E2:19:18:C3:B9:B8",
            "C6:21:3B:AE:36:45",
            "C0:23:61:37:E1:2A",
            "C2:5B:0A:92:20:41",
            "E8:94:83:95:61:E3",
            "C3:79:28:8C:7B:D7"
        ]
        
        # Khởi tạo kết nối database
        self.conn = psycopg2.connect(
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        
    def setup_database(self):
        """Tạo bảng và extension cần thiết trong PostgreSQL"""
        with self.conn.cursor() as cur:
            # Tạo extension pgvector
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Tạo bảng locations
            cur.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    id SERIAL PRIMARY KEY,
                    vector vector(6),
                    label TEXT
                );
            """)
            
            # Tạo index cho vector
            cur.execute("""
                CREATE INDEX IF NOT EXISTS locations_vector_idx 
                ON locations 
                USING ivfflat (vector vector_cosine_ops)
                WITH (lists = 100);
            """)
            
        self.conn.commit()
        print("Database setup completed")
        
    def load_processed_data(self, processed_file):
        """Load dữ liệu đã xử lý từ file JSON"""
        with open(processed_file, 'r') as f:
            data = json.load(f)
            
        return np.array(data['X_scaled']), np.array(data['y'])
        
    def upload_to_database(self, X_scaled, y):
        """Upload dữ liệu đã chuẩn hóa vào PostgreSQL"""
        # Chuẩn bị dữ liệu
        data = [(vector.tolist(), str(label)) for vector, label in zip(X_scaled, y)]
        
        # Upload dữ liệu
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO locations (vector, label)
                VALUES %s
                """,
                data,
                template="(%s::vector, %s)"
            )
            
        self.conn.commit()
        print(f"Uploaded {len(data)} points to PostgreSQL")
        
    def predict_location(self, current_rssi_data, k=5, distance_threshold=10.0):
        """
        Dự đoán vị trí dựa trên dữ liệu RSSI hiện tại
        
        Args:
            current_rssi_data (dict): Dictionary chứa RSSI của các beacon
            k (int): Số lượng neighbors cho KNN
            distance_threshold (float): Ngưỡng khoảng cách tối đa
            
        Returns:
            tuple: (predicted_label, confidence, distances)
        """
        # Chuyển đổi dữ liệu RSSI thành vector
        query_vector = []
        for beacon in self.beacon_order:
            if beacon in current_rssi_data:
                query_vector.append(current_rssi_data[beacon])
            else:
                # Nếu thiếu beacon, sử dụng giá trị trung bình của beacon đó
                query_vector.append(-100)  # hoặc giá trị khác phù hợp
                
        # Tìm k neighbors gần nhất
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT label, 1 - (vector <=> %s::vector) as distance
                FROM locations
                ORDER BY vector <=> %s::vector
                LIMIT %s;
            """, (query_vector, query_vector, k))
            
            results = cur.fetchall()
        
        # Lọc kết quả theo ngưỡng khoảng cách
        valid_results = [(label, distance) for label, distance in results if distance <= distance_threshold]
        
        # Nếu không có kết quả nào thỏa mãn điều kiện khoảng cách,
        # sử dụng kết quả gần nhất
        if not valid_results and results:
            valid_results = [results[0]]
            
        if not valid_results:
            return None, 0, []
            
        # Tính trọng số cho mỗi kết quả dựa trên khoảng cách
        # Sử dụng hàm trọng số nghịch đảo của khoảng cách
        weights = {}
        min_distances = {}  # Lưu khoảng cách nhỏ nhất cho mỗi label
        for label, distance in valid_results:
            # Thêm epsilon nhỏ để tránh chia cho 0
            weight = 1.0 / (distance + 1e-10)
            weights[label] = weights.get(label, 0) + weight
            
            # Cập nhật khoảng cách nhỏ nhất cho label
            if label not in min_distances or distance < min_distances[label]:
                min_distances[label] = distance
            
        # Tìm label có tổng trọng số cao nhất
        max_weight = max(weights.values())
        most_common_labels = [label for label, weight in weights.items() 
                            if weight == max_weight]
        
        # Nếu có nhiều label có cùng trọng số cao nhất,
        # chọn label có khoảng cách nhỏ nhất
        if len(most_common_labels) > 1:
            predicted_label = min(most_common_labels, key=lambda x: min_distances[x])
        else:
            predicted_label = most_common_labels[0]
        
        # Tính độ tin cậy dựa trên tỷ lệ trọng số
        total_weight = sum(weights.values())
        confidence = weights[predicted_label] / total_weight
        
        # Lấy danh sách khoảng cách
        distances = [distance for _, distance in valid_results]
        
        return predicted_label, confidence, distances

def evaluate_predictions(system, test_samples, k=5, distance_threshold=2.0):
    """Đánh giá độ chính xác của hệ thống trên tập test"""
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    print(f"\nEvaluating {len(test_samples)} test samples...")
    
    for i, sample in enumerate(test_samples):
        rssi_values = sample['rssi_values']
        true_label = sample['true_label']
        
        predicted_label, confidence, distances = system.predict_location(
            rssi_values,
            k=k,
            distance_threshold=distance_threshold
        )
        
        if predicted_label is not None:
            total_predictions += 1
            is_correct = str(predicted_label) == str(true_label)
            if is_correct:
                correct_predictions += 1
                
            results.append({
                'sample_id': i,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'distances': distances,
                'is_correct': is_correct
            })
            
            print(f"\nSample {i}:")
            print(f"True label: {true_label}")
            print(f"Predicted label: {predicted_label}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Distances to neighbors: {distances}")
            print(f"Correct: {is_correct}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.2%}")
    print(f"Total predictions: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    
    return results

def main():
    # Khởi tạo hệ thống
    system = IndoorLocationSystemPGVector()
    
    # Tạo database
    system.setup_database()
    
    # Load dữ liệu đã xử lý
    X_scaled, y = system.load_processed_data('processed_data.json')
    system.upload_to_database(X_scaled, y)
    
    # Load và đánh giá các mẫu test
    with open('test_samples.json', 'r') as f:
        test_samples = json.load(f)
    
    results = evaluate_predictions(system, test_samples)

if __name__ == "__main__":
    main() 