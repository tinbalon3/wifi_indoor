import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from collections import Counter

class IndoorLocationSystem:
    def __init__(self, collection_name="indoor_locations"):
        self.collection_name = collection_name
        self.client = QdrantClient("localhost", port=6333)
        self.num_beacons = None
        self.beacon_order = None  # Store beacon order from training data
        self.setup_collection()
        
    def setup_collection(self):
        """Setup collection for storing location data"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            # Create collection with default size, will be updated when data is added
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=10, distance=Distance.COSINE)  # Default size
            )
            
    def create_collection(self):
        """Create a new collection with correct vector size"""
        # Delete collection if it exists
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except:
            pass
            
        # Create new collection with correct size
        if not self.num_beacons:
            raise ValueError("Vector dimension must be set before creating collection")
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.num_beacons, distance=Distance.COSINE)
        )
        print(f"Created collection with vector size: {self.num_beacons}")
        
    def load_processed_data(self, processed_file):
        """Load dữ liệu đã xử lý từ file JSON"""
        with open(processed_file, 'r') as f:
            data = json.load(f)
            
        # Get vector dimension from data
        if len(data['X_scaled']) > 0:
            self.num_beacons = len(data['X_scaled'][0])
        else:
            raise ValueError("No data found in processed file")
            
        # Get beacon order from data
        if 'beacon_order' in data:
            self.beacon_order = data['beacon_order']
        else:
            # If no beacon_order in data, use default order from test data
            self.beacon_order = [
                "C6:21:3B:AE:36:45",
                "E2:19:18:C3:B9:B8",
                "F8:0F:43:E2:80:45",
                "C2:5B:0A:92:20:41",
                "C3:79:28:8C:7B:D7",
                "EA:67:8D:31:27:71",
                "E6:61:40:67:01:E5",
                "CE:88:10:2C:58:A2",
                "C0:23:61:37:E1:2A",
                "E8:94:83:95:61:E3"
            ]
            
        return data['X_scaled'], data['y']
        
    def upload_to_qdrant(self, X_scaled, y):
        """Upload dữ liệu đã chuẩn hóa vào Qdrant"""
        if not self.beacon_order:
            raise ValueError("Beacon order must be set before uploading data")
            
        if not self.num_beacons:
            raise ValueError("Vector dimension must be set before uploading data")
            
        # Create collection with correct size
        self.create_collection()
        
        points = []
        for i, (vector, label) in enumerate(zip(X_scaled, y)):
            # Vector is already in correct order
            points.append(PointStruct(
                id=i,
                vector=vector,
                payload={
                    "label": str(label)
                }
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Uploaded {len(points)} points to Qdrant")
        print(f"Using beacon order: {self.beacon_order}")
        print(f"Vector dimension: {self.num_beacons}")
        
    def add_location_data(self, data):
        """Add location data to the collection"""
        # Determine number of beacons from data
        if self.num_beacons is None:
            # Get the first row's RSSI values to determine number of beacons
            first_row = data.iloc[0]
            rssi_columns = [col for col in first_row.index if col.startswith('RSSI')]
            self.num_beacons = len(rssi_columns)
            
            # Create collection with correct vector size
            self.create_collection()
            
        points = []
        for idx, row in data.iterrows():
            # Get all RSSI values
            rssi_values = [row[col] for col in data.columns if col.startswith('RSSI')]
            point = PointStruct(
                id=idx,
                vector=rssi_values,
                payload={"label": row["Label"]}
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
    def predict_location(self, rssi_values, k=1, distance_threshold=5.0):
        """Predict location based on RSSI values"""
        if not self.beacon_order:
            raise ValueError("Beacon order must be set before prediction")
            
        if len(rssi_values) != self.num_beacons:
            raise ValueError(f"Expected {self.num_beacons} RSSI values, got {len(rssi_values)}")
            
        # Query the collection
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=rssi_values,
            limit=k
        )
        
        if not search_result:
            return None, 0.0, []
            
        # Get distances and labels
        distances = [hit.score for hit in search_result]
        labels = [hit.payload["label"] for hit in search_result]
        
        # Check if any distance is below threshold
        if min(distances) > distance_threshold:
            return None, 0.0, distances
            
        # Calculate weighted prediction
        total_weight = 0
        weighted_label = 0
        
        for distance, label in zip(distances, labels):
            weight = 1.0 / (distance + 1e-10)
            total_weight += weight
            weighted_label += weight * int(label)
            
        predicted_label = round(weighted_label / total_weight)
        confidence = 1.0 / (1.0 + min(distances))
        
        return predicted_label, confidence, distances

def evaluate_predictions(system, test_samples, k=5, distance_threshold=5.0):
    """Đánh giá độ chính xác của hệ thống trên tập test"""
    if not system.beacon_order:
        raise ValueError("Beacon order must be set before evaluation")
        
    correct_predictions = 0
    total_predictions = 0
    total_confidence = 0
    total_distance = 0
    detailed_results = []
    
    print(f"\nEvaluating {len(test_samples)} test samples...")
    print(f"Using beacon order: {system.beacon_order}")
    
    for i, sample in enumerate(test_samples):
        # Convert RSSI dictionary to list using exact beacon order
        rssi_dict = sample['rssi_values']
        rssi_values = [rssi_dict[beacon] for beacon in system.beacon_order]
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
                
            total_confidence += confidence
            total_distance += sum(distances) / len(distances) if distances else 0
            
            detailed_results.append({
                'sample_id': i,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'distances': distances,
                'is_correct': is_correct,
                'rssi_values': rssi_dict
            })
            
            print(f"\nSample {i}:")
            print(f"True label: {true_label}")
            print(f"Predicted label: {predicted_label}")
            print(f"Confidence: {confidence:.2f}")
            print(f"Distances to neighbors: {distances}")
            print(f"Correct: {is_correct}")
            
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0
    avg_distance = total_distance / total_predictions if total_predictions > 0 else 0
    
    summary_results = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'avg_distance': avg_distance,
        'parameters': {
            'k': k,
            'distance_threshold': distance_threshold
        },
        'beacon_order': system.beacon_order  # Include beacon order in results
    }
    
    results = {
        'summary': summary_results,
        'detailed_results': detailed_results
    }
    
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return summary_results

def main():
    # Khởi tạo hệ thống
    system = IndoorLocationSystem()
    
    # Load dữ liệu đã xử lý
    X_scaled, y = system.load_processed_data("processed_data.json")
    system.upload_to_qdrant(X_scaled, y)
    
    # Load và đánh giá các mẫu test
    with open('test_samples.json', 'r') as f:
        test_samples = json.load(f)
    
    # Đánh giá hệ thống
    print("\nEvaluating 300 test samples...")
    results = evaluate_predictions(system, test_samples, distance_threshold=5.0, k=1)
    
    # In kết quả
    print("\nPrediction Results:")
    print(f"Total predictions: {results['total_predictions']}")
    print(f"Correct predictions: {results['correct_predictions']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Average confidence: {results['avg_confidence']:.2%}")
    print(f"Average distance: {results['avg_distance']:.2f}")
    print("\nDetailed results have been saved to 'prediction_results.json'")

if __name__ == "__main__":
    main() 