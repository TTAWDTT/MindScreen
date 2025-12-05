import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. 数据读取与清洗
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'mental_health_social_media_dataset.csv')
df = pd.read_csv(file_path)

# 删除作弊列和无关列
drop_columns = ['person_name', 'date', 'stress_level', 'anxiety_level', 'mood_level']
df_clean = df.drop(drop_columns, axis=1)

# 特征与标签分离
X = df_clean.drop('mental_state', axis=1)
y = df_clean['mental_state']

# 编码
X = pd.get_dummies(X, columns=['gender', 'platform'], drop_first=True)
le = LabelEncoder()
y = le.fit_transform(y)
target_names = le.classes_

# 标准化 (对 PCA 和 SVM 至关重要)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. 高端操作：PCA 降维可视化 (对应第19讲)
# 这张图放在 PPT 前面，展示数据分布
# ==========================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('PCA Projection of Mental Health Data (2D Analysis)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Mental State')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print("PCA 解释方差比:", pca.explained_variance_ratio_)

# ==========================================
# 3. 划分数据
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# 4. 高端操作：网格搜索寻找最优 SVM (对应第15讲附件)
# 展示你不仅会用模型，还会调优
# ==========================================
print("\n正在进行 SVM 超参数网格搜索...")
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid.fit(X_train, y_train)

print(f"SVM 最佳参数: {grid.best_params_}")
best_svm = grid.best_estimator_
svm_pred = best_svm.predict(X_test)

# ==========================================
# 5. 决策树与其特征重要性 (可解释性分析)
# ==========================================
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# ==========================================
# 6. 高端展示：混淆矩阵热力图
# ==========================================
print("\n----- 最终模型评估 (SVM) -----")
print(f"SVM 准确率: {accuracy_score(y_test, svm_pred):.4f}")
print(classification_report(y_test, svm_pred, target_names=target_names, zero_division=0))

# 画混淆矩阵
cm = confusion_matrix(y_test, svm_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - SVM Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 画特征重要性 (决策树)
importances = dt_model.feature_importances_
feature_names = pd.get_dummies(df_clean.drop('mental_state', axis=1), columns=['gender', 'platform'], drop_first=True).columns
feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='magma') # 只取前10个重要特征
plt.title('Top 10 Key Factors Influencing Mental Health')
plt.tight_layout()
plt.show()