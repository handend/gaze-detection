import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# 1. Veriyi oku
df = pd.read_csv("gaze_data.csv")

# 2. Gözler arası mesafeyi hesapla (özellik mühendisliği)
df['eye_distance'] = ((df['right_x'] - df['left_x'])**2 + (df['right_y'] - df['left_y'])**2)**0.5

# 3. Giriş (X) ve çıkış (y) verilerini ayır
X = df[['left_x', 'left_y', 'right_x', 'right_y', 'screen_distance', 'pupil_distance', 'eye_distance']]
y = df[['target_x', 'target_y']]

# 4. Eğitim/test verisi ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modeli oluştur
base_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model = MultiOutputRegressor(base_model)

# 6. Modeli eğit
model.fit(X_train, y_train)

# 7. Test seti ile doğruluk ölç
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Ortalama Kare Hata (MSE): {mse:.4f}")

# 8. Modeli kaydet
joblib.dump(model, "gaze_model.pkl")
print("Model başarıyla kaydedildi.")
