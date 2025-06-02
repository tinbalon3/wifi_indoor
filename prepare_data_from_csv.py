import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np

# Đọc dữ liệu đã chuẩn hóa từ file CSV
# Giả sử cột label tên là 'label'
df = pd.read_csv('data.csv')

X = df.drop('label', axis=1)
y = df['label']

# Chia dữ liệu thành tập train và test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=36, stratify=y if len(y.unique()) > 1 else None)  # 20% cho test set

print(f"Số lượng mẫu huấn luyện: {len(X_train)}")
print(f"Số lượng mẫu test: {len(X_test)}")

# Lưu dữ liệu huấn luyện đã chuẩn hóa
processed_data = {
    'X_scaled': X_train.values.tolist(),
    'y': y_train.tolist(),
    'scaler_mean': [0]*X.shape[1],  # placeholder, vì đã chuẩn hóa sẵn
    'scaler_scale': [1]*X.shape[1], # placeholder, vì đã chuẩn hóa sẵn
    'feature_names': X.columns.tolist()
}
with open('processed_data.json', 'w') as f:
    json.dump(processed_data, f)
print("Đã lưu dữ liệu huấn luyện vào processed_data.json")

# Lưu 300 mẫu test
test_samples = []
for i in range(len(X_test)):
    sample = {
        'rssi_values': {k: float(v) for k, v in X_test.iloc[i].to_dict().items()},
        'true_label': int(y_test.iloc[i])
    }
    test_samples.append(sample)
with open('test_samples.json', 'w') as f:
    json.dump(test_samples, f)
print(f"Đã lưu {len(test_samples)} mẫu test vào test_samples.json") 