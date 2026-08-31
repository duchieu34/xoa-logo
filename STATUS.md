# Trạng thái Project

## Phương pháp tốt nhất hiện tại

Chưa có phương pháp nào đạt Definition of Done.

* TELEA xóa logo mạnh nhưng phá DNA, blur/smear và flicker.
* Alpha deblend giữ texture ở subset đáng tin cậy nhưng chỉ có model cho 25/326 pixel mask.
* Direct temporal chỉ cứu 4 alpha-gap pixel-frame.
* DIS Optical Flow cứu 5 pixel-frame thuộc 2/301 vị trí unresolved tĩnh, nhưng không pixel nào nằm trong lõi logo.

Không dùng GPU/CUDA, inpaint cho output, blur hoặc patch để che phần thất bại.

## Milestone hiện tại

**Experiment 4 đã hoàn tất trên benchmark 192 frame.**

Pipeline:

```text
alpha → direct temporal → DIS Optical Flow → unresolved
```

Flow chạy CPU trên ROI 96×76, dùng donor `t±1..3`, forward/backward consistency, source-clean gate, context màu/gradient và confidence. Vector trong mask được suy ra từ flow context để tránh watermark cố định làm flow thiên về zero.

Farneback đã được thử nhưng cứu 0 pixel unresolved tĩnh. DIS được chọn cho output diagnostic vì tìm được donor spatial thật dưới cùng nguyên tắc an toàn.

## Coverage

* 57 candidate DIS qua clean-source, cycle và context trước confidence;
* 5 donor pixel-frame được chấp nhận;
* 4/192 frame có donor;
* 2/301 vị trí unresolved tĩnh được cứu ít nhất một lần;
* 0 donor nằm trong 186 pixel lõi logo;
* 299 pixel luôn unresolved;
* còn 58.317 unresolved pixel-frame.

Thử thêm `t±6` không tạo donor mới.

## Output diagnostic

```text
outputs/experiment4/ft-vid-23_optical_flow.mp4
diagnostics/experiment4/
```

Video giữ 1920×1080, 24 FPS, 192 frame, 8 giây và audio AAC gốc. Audio SHA-256 trùng input. Diagnostics gồm donor-source, flow, confidence, forward/backward error, unresolved mask, comparison 8 frame và frame DNA khó 128–131.

## Đánh giá chất lượng

Logo-likeness không cải thiện ở độ chính xác 6 chữ số. Không thấy ghosting/double-edge dạng mảng vì coverage chỉ 5 pixel-frame, nhưng donor một phía xuất hiện ngắt quãng ở frame 53–56 tạo nguy cơ sparkle đơn pixel. Temporal MAD tăng nhẹ từ `15,791813` lên `15,792391`.

DNA/đường sáng tại frame 128–131 có flow khoảng 6–8 pixel nhưng không donor nào vượt qua toàn bộ gate. Nới gate ở đây có nguy cơ lấy nhầm nhánh DNA và tạo ghosting, nên không được thực hiện.

## Vấn đề đã biết

* Displacement nhỏ thường chưa đưa tọa độ nguồn ra khỏi watermark.
* Displacement lớn có source sạch nhưng flow/context thường không nhất quán do DNA biến dạng, occlusion và nhiều đường nét gần nhau.
* Không có donor hai phía đồng thuận trong Experiment 4.
* Không có ground truth background để đo PSNR/SSIM tuyệt đối.
* Optical Flow ROI mất khoảng 34,71 giây cho 192 frame, nhanh khoảng 5,53 FPS xử lý nhưng chưa có coverage hữu ích.

## Bước tiếp theo

Chưa nên coi pipeline hiện tại là hybrid đạt yêu cầu. Trước Experiment 5 cần ưu tiên tăng donor lõi có temporal consensus, có thể bằng motion segmentation/occlusion model tốt hơn hoặc donor window thích ứng. Nếu chuyển thẳng sang fallback inpaint, phải ghi rõ đó là đánh đổi chất lượng vì Experiment 1 đã chứng minh DNA bị đứt, smear và flicker.

Experiment 5 **chưa được triển khai**.
