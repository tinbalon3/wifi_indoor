import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu từ file JSON
with open('data.json', 'r') as f:
    data = json.load(f)

# Chuyển đổi dữ liệu thành DataFrame
rows = []
for item in data:
    row = item['beacon'].copy()
    row['label'] = item['label']
    rows.append(row)

df = pd.DataFrame(rows)

# Loại bỏ các mẫu thiếu beacon
df = df.dropna()
print(f"Số lượng mẫu sau khi loại bỏ thiếu beacon: {len(df)}")

# Tách features và labels
X = df.drop('label', axis=1)
y = df['label']

# Tách dữ liệu thành tập huấn luyện và tập test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Chọn ngẫu nhiên 100 mẫu test
test_indices = np.random.choice(len(X_test), size=100, replace=False)
X_test = X_test.iloc[test_indices]
y_test = y_test.iloc[test_indices]

print(f"\nSố lượng mẫu huấn luyện: {len(X_train)}")
print(f"Số lượng mẫu test: {len(X_test)}")

# Phát hiện và loại bỏ outliers chỉ trên tập huấn luyện
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(X_train)
X_train_clean = X_train[outlier_labels == 1]
y_train_clean = y_train[outlier_labels == 1]

print(f"\nSố lượng mẫu huấn luyện sau khi loại bỏ outliers: {len(X_train_clean)}")
print(f"Số lượng outliers bị loại bỏ: {len(X_train) - len(X_train_clean)}")

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)  # Chuẩn hóa tập test với scaler đã fit trên tập huấn luyện

# Thực hiện PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)  # Transform tập test với PCA đã fit trên tập huấn luyện

# Tính toán tỷ lệ phương sai được giải thích
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Vẽ biểu đồ tỷ lệ phương sai được giải thích
plt.figure(figsize=(15, 5))

# Biểu đồ tỷ lệ phương sai cho từng thành phần
plt.subplot(1, 3, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Component')

# Biểu đồ tỷ lệ phương sai tích lũy
plt.subplot(1, 3, 2)
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio')

# Vẽ biểu đồ phân tán của 2 thành phần chính đầu tiên với màu sắc theo label
plt.subplot(1, 3, 3)
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_clean.astype(int), cmap='viridis')
plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance explained)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance explained)')
plt.title('PCA: First Two Principal Components')
plt.colorbar(scatter, label='Label')

plt.tight_layout()
plt.savefig('pca_analysis_results.png', dpi=300, bbox_inches='tight')
plt.close()

# In thông tin chi tiết về các thành phần chính
print("\nThông tin về các thành phần chính:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.2%}")

print("\nTỷ lệ phương sai tích lũy:")
for i, ratio in enumerate(cumulative_variance_ratio):
    print(f"PC{i+1}: {ratio:.2%}")

# Tìm số thành phần cần thiết để giải thích 95% phương sai
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"\nSố thành phần cần thiết để giải thích 95% phương sai: {n_components_95}")

# Lưu dữ liệu đã xử lý
processed_data = {
    'X_scaled': X_train_scaled.tolist(),
    'y': y_train_clean.tolist(),
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'feature_names': X.columns.tolist()
}

with open('processed_data.json', 'w') as f:
    json.dump(processed_data, f)

print("\nĐã lưu dữ liệu đã xử lý vào file processed_data.json")

# Lưu các mẫu test
test_samples = []
for i in range(len(X_test)):
    sample = {
        'rssi_values': X_test.iloc[i].to_dict(),
        'true_label': y_test.iloc[i]
    }
    test_samples.append(sample)

with open('test_samples.json', 'w') as f:
    json.dump(test_samples, f)

print(f"\nĐã lưu {len(test_samples)} mẫu test vào file test_samples.json")

# Phân tích đóng góp của các biến gốc vào các thành phần chính
feature_contributions = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=X.columns
)

print("\nĐóng góp của các biến vào thành phần chính đầu tiên:")
print(feature_contributions['PC1'].sort_values(ascending=False))

# Vẽ biểu đồ nhiệt (heatmap) thể hiện đóng góp của các biến
plt.figure(figsize=(10, 8))
sns.heatmap(feature_contributions, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Contributions to Principal Components')
plt.tight_layout()
plt.savefig('feature_contributions_heatmap.png', dpi=300, bbox_inches='tight')
plt.close() 