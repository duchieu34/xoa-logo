# Trạng thái Project

## Phương pháp tốt nhất hiện tại

**TELEA radius 3 với mask ba ký tự dilation 1 px** là baseline tốt hơn một chút, nhưng chưa đạt chất lượng sử dụng.

TELEA có temporal MAD `10,8845`, thấp hơn Navier–Stokes `12,0186`, và thường ít smear tối hơn nhẹ. Cả hai vẫn làm đứt DNA, mất texture và flicker rõ.

## Milestone hiện tại

**Experiment 1 đã hoàn tất trên benchmark thật.**

Mask:

* sinh từ temporal median của đủ 192 frame và measurement Experiment 0;
* đúng 3 connected component;
* 186 pixel lõi, 326 pixel sau dilation 1 px;
* chỉ chiếm 4,4682% ROI, không dùng bbox chữ nhật đầy;
* overlay đã được kiểm tra trên cả 8 frame đại diện.

Outputs:

```text
outputs/experiment1/ft-vid-23_telea.mp4
outputs/experiment1/ft-vid-23_ns.mp4
```

Cả hai giữ nguyên 1920×1080, 24 FPS, 192 frame, 8 giây và AAC gốc. Hash audio của input và hai output giống hệt nhau.

## Đánh giá chất lượng

| Lỗi | TELEA | Navier–Stokes |
|---|---|---|
| Logo còn nhận ra | Không | Không |
| Frame nền tối/dễ | Hõm tối nhẹ, mềm texture | Tương tự, hơi loang hơn |
| Đường DNA xuyên logo | Bị đứt/smear | Bị đứt/smear, thường tối hơn |
| Blur/mất texture | Có | Có |
| Viền ROI chữ nhật | Không | Không |
| Flicker | Rõ | Rõ, nặng hơn nhẹ |

Worst transition của cả hai là frame 130→131. Radius 1–3 đã được sweep; giảm radius không phục hồi được cấu trúc và radius 1 còn blocky hơn.

## Các vấn đề đã biết

* Inpainting từng frame không có dữ liệu thật để nối lại DNA/texture nằm dưới watermark.
* Patch thay đổi theo background từng frame nên flicker.
* Temporal MAD thấp một phần vì vùng vá bị làm trơn, không phải vì phục hồi đúng.
* Alpha chính xác chưa được ước lượng; Experiment 1 không thực hiện deblending.
* Chưa có optical flow hoặc temporal reconstruction, đúng phạm vi milestone.

## Experiment tiếp theo

Experiment 2 sẽ nghiên cứu alpha watermark/deblending trên cùng mask, ưu tiên dùng thông tin pixel còn lại thay vì đoán toàn bộ vùng bị che. Không thêm optical flow hoặc temporal reconstruction cho đến milestone tương ứng.

Experiment 1 được giữ làm baseline định lượng để so sánh các phương pháp sau.
