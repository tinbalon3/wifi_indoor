import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from collections import Counter
import random

class IndoorLocationSystem:
    def __init__(self, collection_name="indoor_locations"):
        self.collection_name = collection_name
        self.scaler = StandardScaler()
        self.beacon_order = [
            "E2:19:18:C3:B9:B8",
            "C6:21:3B:AE:36:45",
            "C0:23:61:37:E1:2A",
            "C2:5B:0A:92:20:41",
            "E8:94:83:95:61:E3",
            "C3:79:28:8C:7B:D7"
        ]
        
        # Khởi tạo Qdrant client
        self.client = QdrantClient("localhost", port=6333)
        
    def setup_collection(self):
        """Tạo collection trong Qdrant nếu chưa tồn tại"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=6, distance=Distance.EUCLID)
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")
            
    def load_processed_data(self, processed_file):
        """Load dữ liệu đã xử lý từ file JSON"""
        with open(processed_file, 'r') as f:
            data = json.load(f)
            
        # Khôi phục scaler
        # self.scaler.mean_ = np.array(data['scaler_mean'])
        # self.scaler.scale_ = np.array(data['scaler_scale'])
        
        return np.array(data['X_scaled']), np.array(data['y'])
        
    def upload_to_qdrant(self, X_scaled, y):
        """Upload dữ liệu đã chuẩn hóa vào Qdrant"""
        # Chuẩn bị vectors và payloads
        vectors = X_scaled.tolist()
        payloads = [{"label": str(label)} for label in y]
        
        # Upload dữ liệu
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=list(range(len(vectors))),
                vectors=vectors,
                payloads=payloads
            )
        )
        print(f"Uploaded {len(vectors)} points to Qdrant")
        
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
                
        # Chuẩn hóa vector
        query_vector = self.scaler.transform([query_vector])[0]
        
        # Tìm k neighbors gần nhất
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k
        )
        
        # Lọc kết quả theo ngưỡng khoảng cách
        valid_results = [r for r in search_result.points if r.score <= distance_threshold]
        
        # Nếu không có kết quả nào thỏa mãn điều kiện khoảng cách,
        # sử dụng kết quả gần nhất
        if not valid_results and search_result.points:
            valid_results = [search_result.points[0]]
            
        if not valid_results:
            return None, 0, []
            
        # Tính trọng số cho mỗi kết quả dựa trên khoảng cách
        # Sử dụng hàm trọng số nghịch đảo của khoảng cách
        weights = {}
        min_distances = {}  # Lưu khoảng cách nhỏ nhất cho mỗi label
        for result in valid_results:
            label = result.payload['label']
            distance = result.score
            
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
        distances = [r.score for r in valid_results]
        
        return predicted_label, confidence, distances

def evaluate_predictions(system, test_samples, k=5, distance_threshold=5.0):
    """Đánh giá độ chính xác của hệ thống trên tập test"""
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    # Giới hạn số lượng mẫu test
    # test_samples = test_samples[:100]
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
    system = IndoorLocationSystem()
    
    # Tạo collection
    system.setup_collection()
    
    # Load dữ liệu đã xử lý
    X_scaled, y = system.load_processed_data('processed_data.json')
    system.upload_to_qdrant(X_scaled, y)
    
    # Load và đánh giá các mẫu test
    with open('test_samples.json', 'r') as f:
        test_samples = json.load(f)
    
    results = evaluate_predictions(system, test_samples,distance_threshold=10.0,k=11)

if __name__ == "__main__":
    main() 