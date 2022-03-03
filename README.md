# VietOCR
**Các bạn vui lòng cập nhật lên version vietocr>=0.3.8 để không xảy ra lỗi.**
<p align="center">
<img src="https://github.com/pbcquoc/vietocr/raw/master/image/sample.png" width="1000" height="300">
</p>

Trong project này, mình cài đặt mô hình Transformer OCR nhận dạng chữ viết tay, chữ đánh máy cho Tiếng Việt. Kiến trúc mô hình là sự kết hợp tuyệt vời giữ mô hình CNN và Transformer (là mô hình nền tảng của BERT khá nổi tiếng). Mô hình TransformerOCR có rất nhiều ưu điểm so với kiến trúc của mô hình CRNN đã được mình cài đặt. Các bạn có thể đọc [tại](https://pbcquoc.github.io/vietocr) đây về kiến trúc và cách huấn luyện mô hình với các tập dữ liệu khác nhau. 

Mô hình VietOCR có tính tổng quát cực tốt, thậm chí có độ chính xác khá cao trên một bộ dataset mới mặc dù mô hình chưa được huấn luyện bao giờ. 

<p align="center">
<img src="https://raw.githubusercontent.com/pbcquoc/vietocr/master/image/vietocr.jpg" width="512" height="614">
</p>

# Cài Đặt
Để cài đặt các bạn gõ lệnh sau
```
pip install vietocr
```
# Quick Start
Các bạn tham khảo notebook [này](https://github.com/pbcquoc/vietocr/blob/master/vietocr_gettingstart.ipynb) để biết cách sử dụng nhé. 

# Model Zoo 
Thư viện này cài đặt cả 2 kiểu seq model đó là attention seq2seq và transfomer. Seq2seq có tốc độ dự đoán rất nhanh và được dùng trong industry khá nhiều, tuy nhiên transformer lại chính xác hơn nhưng lúc dự đoán lại khá chậm. Do đó mình cung cấp cả 2 loại cho các bạn lựa chọn. 

Mô hình này được huấn luyện trên tập dữ liệu gồm 10m ảnh, bao gồm nhiều loại ảnh khác nhau như ảnh tự phát sinh, chữ viết tay, các văn bản scan thực tế. 
Pretrain model được cung cấp sẵn.

# Kết quả thử nghiệm trên tập 10m
| Backbone         | Config           | Precision full sequence | time |
| ------------- |:-------------:| ---:|---:|
| VGG19-bn - Transformer | vgg_transformer | 0.8800 | 86ms @ 1080ti  |
| VGG19-bn - Seq2Seq     | vgg_seq2seq     | 0.8701 | 12ms @ 1080ti |

Thời gian dự đoán của mô hình vgg-transformer quá lâu so với mô hình seq2seq, trong khi đó không có sự khác biệt rõ ràng giữ độ chính xác của 2 loại kiến trúc này.

# Dataset 
Mình chỉ cung cấp tập dữ liệu mẫu khoảng 1m ảnh tự phát sinh. Các bạn có thể tải về tại [đây](https://drive.google.com/file/d/1T0cmkhTgu3ahyMIwGZeby612RpVdDxOR/view).
# License
Mình phát hành thư viện này dưới các điều khoản của [Apache 2.0 license]().

# Liên hệ
Nếu bạn có bất kì vấn đề gì, vui lòng tạo issue hoặc liên hệ mình tại pbcquoc@gmail.com 
