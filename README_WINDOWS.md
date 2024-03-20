# Hướng dẫn Fix Lỗi Thực Thi trên Windows với Python 11

### 1. Sử dụng Python thay vì các lệnh wget và unzip

`Lỗi: Không dùng được wget và unzip`

**Data sample**

```
! wget https://vocr.vn/data/vietocr/sample.zip
! unzip  -qq -o sample.zip
```

**Chuyển sang dùng python**

```python
import requests
import zipfile
import io

# Download the ZIP file
url = "https://vocr.vn/data/vietocr/sample.zip"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Extract the contents of the ZIP file
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Failed to download the ZIP file.")
```

**check sameple file**

```
# ! ls sample | shuf |head -n 5
```

**Chuyển sang dùng python**

```python
import os

# List files in the 'sample' directory
files = os.listdir("sample")

# Select the first 5 files
selected_files = files[:5]

# Print the selected files
print(selected_files)
```

**Data line**

```
! wget https://vocr.vn/data/vietocr/data_line.zip
! unzip -qq -o ./data_line.zip
```

**Chuyển sang dùng python**

```python
import requests
import zipfile
import io

# Download the ZIP file
url = "https://vocr.vn/data/vietocr/data_line.zip"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Extract the contents of the ZIP file
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as zip_ref:
        zip_ref.extractall(".")
else:
    print("Failed to download the ZIP file.")
```

### 2. Sửa lỗi phần Train mode

- `Lỗi Cannot pickle 'Environment' object`: Bạn cập nhật config để sử dụng num_workers là 0. Vì đối với windows num_workers > 0 có thể có thể gây ra sự cố, có thể là lỗi xử lý hoặc hiệu suất chậm. Bạn có thể đọc thêm thông tin tại đây [Errors when using num workers 0 in dataloader](https://discuss.pytorch.org/t/errors-when-using-num-workers-0-in-dataloader/97564/5)

```
config['dataloader']['num_workers'] = 0
```

- `Lỗi: Không đủ dung lượng trên ổ đĩa` khi chạy

```python
trainer = Trainer(config, pretrained=True)
```

```bash
train_hw: There is not enough space on the disk
valid_hw: There is not enough space on the disk
```

vì trong setup tạo [dataset](vietocr\tool\create_dataset.py) bộ `train` và `valid` được đặt với kích thước 1 Terabyte.

```python
env = lmdb.open(outputPath, map_size=1099511627776)
```

- Xác định dung lượng trống trên ổ đĩa và điều chỉnh kích thước bộ train và valid:

- Ví dụ: Máy mình còn 150G trống. Mình set up thành 60G cho mỗi train và valid.

```python
env = lmdb.open(outputPath, map_size=64424509440)  # Thay đổi kích thước LMDB theo dung lượng ổ đĩa trống
```

- Lỗi liên quan đến mã hóa và kiểu dữ liệu trong file [create_dataset](vietocr\tool\create_dataset.py) và imgaug library.

```python
trainer.visualize_dataset()
```

- Lỗi `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8f in position 136: character maps to <undefined>`
  Dòng 45 file tool/create_dataset.py
  Bạn sửa

```
open(annotation_path, 'r')
```

thành

```
open(annotation_path, 'r', encoding='utf-8')
```

- Lỗi `UnicodeEncodeError: 'charmap' codec can't encode characters in position 4-5: character maps to <undefined>`
  Bạn sửa

```
open(fname_path, 'w')
```

thành

```
open(fname_path, 'w', encoding='utf-8')
```

2 lỗi trên là do Python sử dụng mã hóa mặc định của hệ thống. Và tất nhiên không phải utf-8 rùi.

- Lỗi `AttributeError: module 'numpy' has no attribute 'bool'.` Đây là do từ `NumPy 1.24` thì `np.bool` không được dùng nữa và loại bỏ hoàn toàn thay vào đó được kê thừa thành `np.bool_`. Nhưng thư viện imgaug vẫn sử dụng. Bạn vào và tại [meta.py](\imgaug\augmenters\meta.py) dòng 3368 sửa `dtype=np.bool` thành `dtype=np.bool_` .
  Bạn có thể xem chi tiết tại [1.20.0-notes](https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations)

# **Trên đây là hướng dẫn fix những lỗi gặp phải khi mình thực hiện. Cảm ơn các bạn đã xem.**

**Note**:
Bạn có thể tìm tất cả `'r') ` sửa thành `'r', encoding='utf-8')`

Có một lỗi nữa là
`AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'`
trong `site-packages\vietocr\tool\translate.py` dòng 149
sửa

```
img = img.resize((new_w, image_height), Image.ANTIALIAS)
```

thành

```
img = img.resize((new_w, image_height), Image.BILINEAR)
```
