# VietOCR
Trong project này, mình cài đặt mô hình Transformer OCR nhận dạng chữ viết tay, chữ đánh máy cho Tiếng Việt. Kiến trúc mô hình là sự kết hợp tuyệt vời giữ mô hình CNN và Transformer (là mô hình nền tảng của BERT khá nổi tiếng). Mô hình TransformerOCR có rất nhiều ưu điểm so với kiến trúc của mô hình CRNN đã được mình cài đặt. Các bạn có thể đọc [tại]() đây về kiến trúc và cách huấn luyện mô hình với các tập dữ liệu khác nhau. 

Mô hình VietOCR có tính tổng quát cực tốt, thậm chí có độ chính xác khá cao trên một bộ dataset mới mặc dù mô hình chưa được huấn luyện bao giờ. 

<p align="center">
<img src="https://raw.githubusercontent.com/pbcquoc/vietocr/master/image/vietocr.jpg" width="512" height="614">
</p>

# Cài đặt
Để cài đặt các bạn gõ lệnh sau
```
pip install vietocr
```
# Quick Start
Các bạn tham khảo notebook [này](https://colab.research.google.com/drive/1GVRKGFPsmq_xKJbOuSplrIDJZYp-IyzW?usp=sharing) để biết cách sử dụng nhé. 

# Model zoo 
Mô hình này được huấn luyện trên tập dữ liệu gồm 10m ảnh, bao gồm nhiều loại ảnh khác nhau như ảnh tự phát sinh, chữ viết tay, các văn bản scan thực tế. 
Pretrain model các bạn download tại [đây]().

Đồng thời mình cũng thử nghiệm kết quả của mô hình trên tập dữ liệu [synth 90k](https://www.robots.ox.ac.uk/~vgg/data/text/) mô hình transformerocr cho có độ chính xác full_sequence là 96% trong khi đó một cài đặt khá phổ biến khác dữ trên cơ chế attention cho kết quả là 93%.

# License
Mình phát hành thư viện này dưới các điều khoản của [Apache 2.0 license]().

# Liên hệ
Nếu bạn có bất kì vấn đề gì, vui lòng tạo issue hoặc liên hệ mình tại pbcquoc@gmail.com 

(to be continued)
