# Trạng thái Project

## Phương pháp tốt nhất hiện tại

Chưa có phương pháp xóa. Experiment 0 chỉ khảo sát và chủ đích không thay đổi pixel nào.

## Milestone hiện tại

**Experiment 0 đã hoàn tất trên benchmark thật.**

Benchmark `samples/ft-vid-23.mp4`:

* H.264 `yuv420p`, 1920×1080, 24 FPS;
* 192 frame, thời lượng 8 giây;
* audio AAC có mặt và cùng thời lượng;
* đã xuất 8 frame đại diện và đầy đủ diagnostics;
* 6 unit test và smoke test end-to-end đều đạt.

Watermark đo được:

* bbox phần chữ xấp xỉ `(1864, 1042, 32, 14)` ở 1080p;
* cách mép phải/dưới khoảng 24 px;
* ROI context mặc định `(0.95, 0.93, 0.05, 0.07)`, tương ứng `(1824, 1004, 96, 76)`;
* màu đích gần trắng, có alpha/transparency và viền anti-alias;
* cố định theo tọa độ frame, sai số phép đo ≤1 px.

## Các vấn đề đã biết

* Alpha chính xác chưa thể xác định duy nhất từ video đã compositing; cần ước lượng trong Experiment 2.
* Mask hiện chưa tồn tại. Bbox đo được không được dùng như mask hình chữ nhật để inpaint.
* Nhiều frame có DNA, thanh màu và đường sáng chạy trực tiếp qua logo; baseline theo từng frame nhiều khả năng làm đứt nét hoặc flicker.
* Chưa có video output hay đánh giá phục hồi liên tục vì chưa bắt đầu phương pháp xóa.
* OpenCV được cài trong `.venv` Python 3.11, không dựa vào Python hệ thống.

## Experiment tiếp theo

Experiment 1 sẽ:

1. tạo mask bám ba ký tự từ temporal evidence, có dilation rất nhỏ chỉ cho viền anti-alias;
2. lưu mask và overlay để kiểm tra trước khi xóa;
3. chạy hai baseline CPU trên cùng mask: `cv2.INPAINT_TELEA` và `cv2.INPAINT_NS`;
4. xuất ảnh so sánh trên cả frame dễ và khó;
5. encode hai video giữ nguyên 1920×1080, 24 FPS và mux lại audio AAC gốc;
6. đánh giá flicker, smear, đường DNA bị đứt và thời gian xử lý.

Không dùng blur, crop, patch phẳng, GPU hoặc model AI.
