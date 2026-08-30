# Xóa Watermark Veo — Nhật ký nghiên cứu

Project này được chủ đích phát triển theo **một chuỗi các thử nghiệm có đo lường và đánh giá**.

Thứ tự ưu tiên về chất lượng là:

1. **Độ chính xác khi khôi phục hình ảnh**
2. **Độ ổn định giữa các frame theo thời gian**
3. **Tốc độ xử lý**

Toàn bộ quá trình xử lý phải **chỉ sử dụng CPU**.

## Experiment 0 — Khảo sát video và vùng ROI ứng viên

### Giả thuyết

Trước khi lựa chọn phương pháp xóa watermark, cần lấy các **frame đại diện** và xác định một **ROI không phụ thuộc vào độ phân giải**, nhằm xác định:

* Watermark Veo mặc định có nằm cố định tại cùng một vị trí trong video hay không.
* Các pixel nhìn thấy của watermark thay đổi như thế nào khi background phía dưới thay đổi.
* Việc sử dụng **alpha deblending** để phục hồi phần hình ảnh phía dưới watermark có khả thi hay không.

### Phương pháp

* Sử dụng `ffprobe` để kiểm tra container và các stream của video.
* Chọn **8 frame phân bố đều** trên toàn bộ timeline của video.
* Chuyển ROI được khai báo theo tỷ lệ tương đối thành tọa độ pixel phù hợp với độ phân giải của video nguồn.
* Lưu lại cho mỗi frame:

  * frame gốc;
  * frame có vẽ vùng ROI;
  * vùng ROI được crop nguyên bản, chưa chỉnh sửa.
* Lưu thêm:

  * contact sheet;
  * ảnh ROI trung vị theo thời gian (`temporal median ROI`);
  * ảnh trực quan hóa độ lệch chuẩn của ROI theo thời gian (`temporal standard-deviation`);
  * các thống kê mô tả của từng frame.
* Không chỉnh sửa hoặc encode video.

## Kết quả

Đã chạy Experiment 0 trên `samples/ft-vid-23.mp4`.

### Metadata benchmark

| Thuộc tính | Giá trị |
|---|---:|
| Video | H.264, `yuv420p` |
| Độ phân giải | 1920×1080 |
| FPS | 24 |
| Thời lượng | 8,000 giây |
| Số frame | 192 |
| Audio | AAC, có mặt, thời lượng 8,000 giây |
| Kích thước file | 31.716.249 byte |

### Vị trí và kích thước watermark

Phân tích temporal median trên toàn bộ 192 frame và ngưỡng pixel sáng/ít bão hòa cho ba ký tự cho kết quả:

```text
bbox phần chữ nhìn thấy, tại 1920×1080:
x = 1864
y = 1042
width  ≈ 32 px
height ≈ 14 px
```

Logo cách mép phải và mép dưới khoảng 24 px. Bbox tương đối xấp xỉ `(0.97083, 0.96481, 0.01667, 0.01296)`. Đây là bbox đo từ pixel nhìn thấy, **chưa phải mask alpha cuối cùng**; viền anti-alias có thể mở rộng thêm 1–2 px ở mức alpha thấp.

ROI context mặc định sau hiệu chỉnh là `(0.95, 0.93, 0.05, 0.07)`, tương ứng `(1824, 1004, 96, 76)` ở 1080p. ROI giữ khoảng 24–40 px context quanh logo nhưng chỉ chiếm khoảng 0,35% diện tích frame.

### Màu, alpha và độ cố định

* Logo là ba ký tự sáng, màu đích gần trắng/trung tính.
* Màu đo được bên trong nét chữ thay đổi theo DNA/background phía dưới; temporal median tại các pixel lõi vẫn bị nhuộm màu. Điều này cùng viền anti-alias xác nhận logo **không phải lớp trắng hoàn toàn opaque**.
* Chưa gán một giá trị alpha chính xác trong Experiment 0: chỉ có ảnh đã compositing thì `W`, `alpha` và background `B` chưa xác định duy nhất. Việc ước lượng alpha định lượng thuộc Experiment 2, sau khi có mask và nguồn background tạm ước lượng.
* Tìm dịch chuyển template trong cửa sổ ±3 px trên 192 frame cho kết quả `(0,0)` ở 190 frame; 2 frame chọn `(1,0)` do nền sáng gây nhiễu. Kết luận: logo **cố định theo tọa độ frame**, với sai số đo không quá 1 px; không có bằng chứng logo tự chuyển động.

### Frame đại diện và độ khó

Tám frame đã lưu: `0, 27, 54, 81, 109, 136, 163, 191`.

* Frame 136 là trường hợp tương đối dễ: phần lớn chữ nằm trên vùng tối, nhưng viền DNA vẫn chạm logo.
* Frame 0 và 27 là mức trung bình: đường DNA lớn nằm gần hoặc cắt một phần ký tự.
* Frame 54, 81, 109, 163 và 191 là nhóm khó: thanh màu, đường viền sáng hoặc cấu trúc DNA chạy trực tiếp qua logo.

Benchmark này không phù hợp với giả định “background đơn giản”. Inpainting theo từng frame có nguy cơ làm đứt đường, smear và flicker cao.

Diagnostics cuối nằm trong `diagnostics/experiment0/` gồm frame gốc, overlay, ROI crop, hai contact sheet, temporal median, temporal standard deviation và `report.json`. Thư mục này được giữ local và không commit media vào Git.

Phép đo nhỏ, có checksum SHA-256 của benchmark để truy vết, được lưu tại `assets/veo_1080p_measurement.json`.

### Ưu điểm

* Không đưa ra các giả định có thể làm mất hoặc phá hỏng thông tin của logo ngay từ đầu.
* Hoạt động với nhiều độ phân giải khác nhau nhờ sử dụng tọa độ tương đối.
* Tạo ra dữ liệu cần thiết để lựa chọn các frame khó, có texture phức tạp hoặc có chuyển động để kiểm thử.

### Hạn chế

* Việc lấy frame phân bố đều theo thời gian chỉ là bước khảo sát đầu tiên; nó chưa thể tự hiểu và phân loại độ khó của từng cảnh.
* Ảnh median và ảnh thể hiện sự thay đổi theo thời gian có thể giúp phát hiện các cấu trúc ổn định, nhưng **chúng không phải alpha mask**.
* Chưa thể đánh giá chất lượng phục hồi vì Experiment 0 chủ đích không thay đổi pixel.
* Phép đo bbox dựa trên temporal statistics; mask alpha chính xác vẫn cần Experiment 1–2.

### Quyết định

Ba điều kiện khảo sát đã hoàn tất: benchmark đã chạy, ROI đã hiệu chỉnh và các frame đại diện đã được kiểm tra trực quan. Có thể bắt đầu Experiment 1 trong milestone kế tiếp, nhưng không triển khai lẫn vào commit Experiment 0.

---

## Chiến lược phục hồi đề xuất — Tạm thời

Chiến lược dựa trên dữ liệu thực tế sẽ được quyết định chính thức sau Experiment 0.

Phương án ứng viên hiện tại là:

1. Quan sát watermark qua nhiều frame để ước lượng một **template ổn định về màu sắc và alpha của logo**.
2. Tại những pixel mà việc ước lượng đủ đáng tin cậy, thực hiện **đảo ngược phép compositing** để cố khôi phục background gốc.
3. Lấy thông tin sạch từ cả:

   * các frame trước;
   * các frame sau.
4. Sử dụng **Optical Flow** để warp các thông tin đó về đúng vị trí của frame hiện tại.
5. Thực hiện **forward/backward consistency check** để kiểm tra Optical Flow có đáng tin cậy hay không.
6. Chỉ blend những pixel có độ tin cậy cao.
7. Với những pixel trong mask vẫn chưa thể phục hồi, sử dụng **CPU inpainting với mask bám sát hình dạng watermark** như phương án cuối cùng.
8. Khi xảy ra:

   * **scene cut** — chuyển cảnh;
   * **occlusion** — vật thể bị che khuất;

   thì phải vô hiệu hóa các frame lân cận không còn phù hợp để làm nguồn phục hồi.

Có thể hình dung chiến lược tạm thời như sau:

```text
Quan sát nhiều frame
        ↓
Ước lượng màu + alpha của Veo
        ↓
Alpha deblending
        ↓
Khôi phục được hết?
       /       \
     Có        Không
      │           │
      │           ▼
      │     Frame trước + sau
      │           ↓
      │      Optical Flow
      │           ↓
      │    Warp về frame hiện tại
      │           ↓
      │   Kiểm tra forward/backward
      │           ↓
      │    Chỉ lấy pixel đáng tin
      │           ↓
      │     Vẫn còn pixel thiếu?
      │          /        \
      │       Không       Có
      │         │          │
      │         │     CPU Inpainting
      │         │     vùng nhỏ còn lại
      │         │          │
      └─────────┴──────────┘
                ↓
          ROI đã phục hồi
```

Tóm lại, hướng nghiên cứu hiện tại là **ưu tiên lấy lại thông tin thật trước**, tận dụng thông tin từ các frame lân cận sau đó, và chỉ dùng inpainting để **đoán những pixel cuối cùng không còn cách nào phục hồi được**.
