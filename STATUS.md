# Trạng thái Project

## Phương pháp tốt nhất hiện tại

Chưa có phương pháp nào đạt Definition of Done.

* TELEA xóa logo mạnh nhưng phá DNA, blur/smear và flicker.
* Alpha deblend giữ texture ở subset đáng tin cậy nhưng chỉ có model cho 25/326 pixel mask.
* Direct temporal không Optical Flow chỉ cứu thêm 4 pixel-frame và 0/301 pixel unresolved tĩnh.

Không hạ confidence, không dùng inpaint, blur hay patch để che phần thất bại.

## Milestone hiện tại

**Experiment 3 đã hoàn tất trên toàn bộ benchmark 192 frame.**

Thứ tự xử lý đã triển khai:

```text
alpha-resolved → temporal donor đáng tin cậy → giữ unresolved
```

Donor chỉ đến từ cùng tọa độ ở `t±1..3`, phải alpha-clean tại source và vượt qua kiểm tra context màu/gradient. Không Optical Flow hoặc dịch chuyển không gian.

Coverage chính:

* 301 pixel unresolved tĩnh từ Experiment 2;
* 534 lỗ alpha động theo frame;
* 4 donor pixel-frame được chấp nhận, tại 2 vị trí và 4 frame;
* cứu 0/301 pixel unresolved tĩnh;
* cứu 4/534 lỗ alpha động, tương đương 0,75%;
* còn 58.322 unresolved pixel-frame.

Không có donor hai phía đồng thuận. Hai donor lấy từ `t−1`, hai donor từ `t+1`.

## Output diagnostic

```text
outputs/experiment3/ft-vid-23_direct_temporal.mp4
diagnostics/experiment3/
```

Video giữ 1920×1080, 24 FPS, 192 frame, 8 giây và audio AAC gốc. Audio hash trùng input. Diagnostics có donor-source map, confidence map, unresolved mask, coverage report và comparison cho 8 frame đại diện cùng bốn donor event.

## Đánh giá chất lượng

Direct temporal gần như giống alpha-only về thị giác. Bốn donor không tạo ghosting/flicker nhìn thấy được, nhưng coverage quá nhỏ để giảm logo. DNA/đường sáng chuyển động vẫn unresolved; logo còn rõ. Không có blur, smear hoặc patch mới vì pixel thiếu bằng chứng được giữ nguyên.

Ba nhóm từ chối chính:

* 57.975 pixel-frame không có source alpha-clean;
* 300 pixel-frame sai context;
* 47 pixel-frame không đủ confidence hoặc đồng thuận.

## Các vấn đề đã biết

* 301 pixel unresolved tĩnh không thể trở thành donor ở cùng tọa độ vì không frame nào có alpha recovery sạch tại các pixel đó.
* Background DNA chuyển động làm nội dung sạch xuất hiện ở tọa độ khác giữa các frame.
* Direct copy không có motion compensation nên không thể tái sử dụng thông tin đã dịch chuyển.
* Không có ground truth background để đo PSNR/SSIM phục hồi tuyệt đối.

## Experiment tiếp theo

Experiment 4 nên thử Optical Flow CPU trong ROI để warp frame trước/sau về frame hiện tại, kèm forward/backward consistency, occlusion gate và confidence. Mục tiêu đầu tiên là chứng minh có thể cứu một phần 301 pixel unresolved mà không tạo ghosting/double edge trên DNA chuyển động.

Experiment 4 **chưa được triển khai**.
