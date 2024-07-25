# Dự Án Phát Hiện Ngôn Ngữ
Dự án này sử dụng các kỹ thuật học máy để tự động phát hiện ngôn ngữ của các văn bản được đưa vào. Trình bày ba mô hình phân loại khác nhau: Naive Bayes, Random Forest và Mạng Nơ-ron Tích chập (CNN).
## Mục Lục
- [Tổng Quan Dự Án](#tổng-quan-dự-án)
- [Cài Đặt](#cài-đặt)
- [Tập Dữ Liệu](#tập-dữ-liệu)
- [Các Mô Hình](#các-mô-hình)
- [Naive Bayes](#naive-bayes)
- [Random Forest](#random-forest)
- [CNN](#cnn)
- [Dự Đoán](#dự-đoán)
- [Các Thông Số Đánh Giá](#các-thông-số-đánh-giá)
- [Người Đóng Góp](#người-đóng-góp)
## Tổng Quan Dự Án
Mục tiêu chính của dự án này là phân loại dữ liệu văn bản thành các ngôn ngữ khác nhau bằng cách sử dụng các mô hình học máy khác nhau. Mã nguồn thực hiện việc tiền xử lý văn bản, vector hóa, huấn luyện các mô hình và đánh giá hiệu suất của chúng.
## Cài Đặt
Đảm bảo bạn đã cài đặt Python 3.x cùng với các thư viện sau:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow` (để sử dụng cho CNN)
Bạn có thể cài đặt các gói này bằng pip:
```bash\npip install pandas numpy matplotlib seaborn scikit-learn tensorflow```
## Tập dữ liệu
Tập dữ liệu sử dụng được lấy ở link: https://www.kaggle.com/datasets/basilb2s/language-detection, chứa hai cột chính:
- Text: Dữ liệu văn bản cần phân loại.
- Language: Nhãn ngôn ngữ tương ứng của văn bản.
## Các Mô Hình
### Naive Bayes
- **Tiền xử lý:** Dữ liệu được làm sạch và biến đổi bằng cách sử dụng vector hóa TF-IDF.
- **Huấn luyện Mô Hình:** Sử dụng bộ phân loại `MultinomialNB` với việc tối ưu hóa tham số thông qua `GridSearchCV`.
- **Đánh Giá:** Mô hình được đánh giá bằng các thông số như độ chính xác, độ tinh chuẩn, độ hồi tưởng và điểm F1.
### Random Forest
- **Tiền xử lý:** Kỹ thuật vector hóa tương tự như đã sử dụng trong Naive Bayes.
- **Huấn luyện Mô Hình:** `RandomForestClassifier` được huấn luyện với việc tối ưu hóa tham số thông qua `GridSearchCV`.
- **Đánh Giá:** Hiệu suất được đánh giá bằng các thông số tương tự như trên.
### CNN
- **Xây dựng Mô Hình:** Một CNN đơn giản với các lớp denses được xây dựng cho phân loại ngôn ngữ.
- **Huấn luyện:** Mô hình được biên dịch và huấn luyện với kích thước lô và số lượng epoch xác định.
- **Đánh Giá:** Giá trị mất mát và độ chính xác trên tập kiểm tra được báo cáo.
- ## Dự Đoán
- Các hàm sau đây được cung cấp để thực hiện dự đoán cho các mô hình:
### Đối với Naive Bayes và Random Forest
```python\ndef predict(text):  
text_tfidf = tfidf_vectorizer.transform([text]).toarray()  
lang = best_model.predict(text_tfidf)  
language = encoder.inverse_transform(lang)[0]  
print('Ngôn ngữ là', language)  
```
### Đối với CNN
```python\ndef predict(text):  
text_tfidf = tfidf_vectorizer.transform([text]).toarray()  # Chuyển văn bản sang TF-IDF  
predictions = best_model.predict(text_tfidf)  # Thực hiện dự đoán  
predicted_class = np.argmax(predictions, axis=1)  
language = encoder.inverse_transform(predicted_class)  # Chuyển đổi chỉ số thành nhãn  
return language[0]  
```
## Các Thông Số Đánh Giá
Dự án tính toán các chỉ số đánh giá khác nhau cho các mô hình phân loại, bao gồm:
- Độ chính xác
- Độ tinh chuẩn
- Độ hồi tưởng
- Điểm F1
- Ma trận nhầm lẫn
Các chỉ số này được in ra console để so sánh hiệu suất giữa các mô hình khác nhau.
## Người Đóng Góp
- Nguyễn Duy Vụ
- Trần Thụy Minh Thư
