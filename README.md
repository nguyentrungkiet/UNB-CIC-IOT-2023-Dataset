# UNB-CIC-IOT-2023 Dataset Analysis

## 📊 Mô tả dự án
Dự án phân tích bộ dữ liệu UNB-CIC-IOT-2023 sử dụng nhiều mô hình Machine Learning và Deep Learning để phát hiện các cuộc tấn công mạng IoT.

## 🔬 Các mô hình được sử dụng
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost**
- **LSTM (Long Short-Term Memory)**
- **CNN-LSTM**
- **Mamba**

## 📁 Cấu trúc thư mục
```
├── 2_nhan/          # Phân loại 2 lớp (Binary Classification)
├── 4_nhan/          # Phân loại 4 lớp (Multi-class Classification)
├── 8_nhan/          # Phân loại 8 lớp (Multi-class Classification)
├── anh/             # Hình ảnh confusion matrix
├── bieudo_data/     # Dữ liệu biểu đồ
├── model_comparison_results/  # Kết quả so sánh các mô hình
├── compare_models.py          # Script so sánh mô hình
└── Tach_data.py              # Script xử lý dữ liệu
```

## 🎯 Kết quả
Mỗi thư mục chứa:
- **Source code** của từng mô hình
- **Biểu đồ kết quả** (ROC curves, confusion matrix, learning curves)
- **Metrics** (accuracy, precision, recall, F1-score)
- **Feature importance** và **SHAP values**

## 🚀 Cách sử dụng
1. Clone repository:
   ```bash
   git clone https://github.com/nguyentrungkiet/UNB-CIC-IOT-2023-Dataset.git
   ```

2. Cài đặt dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow xgboost matplotlib seaborn
   ```

3. Chạy các script:
   ```bash
   python 2_nhan/logistic/logistic.py
   python 4_nhan/randomforest/randomforest.py
   # ... các mô hình khác
   ```

## 📈 So sánh mô hình
Chạy script so sánh:
```bash
python compare_models.py
```

## 🤝 Đóng góp
Mọi đóng góp đều được hoan nghênh! Hãy tạo Pull Request hoặc Issue.

## 📧 Liên hệ
- **Author**: Nguyen Trung Kiet
- **Email**: trungkiet1993@gmail.com
- **GitHub**: [@nguyentrungkiet](https://github.com/nguyentrungkiet)

## 📄 License
MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.