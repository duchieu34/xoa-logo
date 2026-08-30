# Trạng thái Project

## Phương pháp tốt nhất hiện tại

Chưa có phương pháp nào đạt Definition of Done.

* **TELEA** xóa logo mạnh hơn nhưng phá DNA, blur/smear và flicker.
* **Alpha deblend only** giữ texture tốt hơn tại subset đáng tin cậy, nhưng chỉ xử lý được 7,67% mask nên logo vẫn rõ.

Không hạ confidence và không dùng inpaint để che 301 pixel unresolved.

## Milestone hiện tại

**Experiment 2 đã hoàn tất trên benchmark thật.**

Mô hình được chọn là temporal-distribution quantile matching:

```text
I = B·(1−alpha) + W·alpha
W ≈ BGR(255,255,255)
```

`W` chạm biên 255 nên chỉ kết luận watermark trắng/gần trắng. Alpha map, confidence map, resolved mask và unresolved mask đã được xuất.

Coverage:

* 25/326 pixel mask resolved — 7,67%;
* 23/186 pixel lõi resolved — 12,37%;
* median mỗi frame thực sự áp dụng 22 pixel, trong đó 20 pixel lõi;
* 301/326 pixel unresolved — 92,33%.

Lý do unresolved:

* 162 pixel không có proxy phân phối hợp lý;
* 128 pixel confidence thấp;
* 11 pixel thất bại ở gamut/gate khác.

## Output diagnostic

```text
outputs/experiment2/ft-vid-23_alpha_deblend_only.mp4
```

Video giữ 1920×1080, 24 FPS, 192 frame, 8 giây và audio AAC gốc. Đây chỉ là artifact nghiên cứu; logo vẫn thấy rõ.

## Đánh giá chất lượng

| Lỗi/thuộc tính | TELEA | Alpha deblend only |
|---|---|---|
| Logo còn nhận ra | Không | Có, gần như nguyên vẹn |
| DNA/texture tại pixel không chắc chắn | Bị đoán và phá | Giữ nguyên input |
| Smear/blur dạng mảng | Nặng ở frame khó | Không |
| Pixel phục hồi có confidence | Không phân biệt | 25 pixel tĩnh |
| Unresolved được ghi rõ | Không | 301 pixel |
| Temporal MAD | 10,8845, thấp do blur | 15,7918; original 15,5847 |
| Artifact chính | Patch tối, đứt DNA, flicker | Logo còn sót, sparkle nhỏ cục bộ |

## Các vấn đề đã biết

* Không có ground truth background, nên `W/alpha/B` không xác định duy nhất.
* Các nét lõi alpha cao làm inverse khuếch đại sai số và thường ra ngoài gamut.
* Background DNA chuyển động khiến spatial plane và same-frame proxy thất bại.
* Distribution matching chịu được lệch pha tốt hơn nhưng vẫn thiếu proxy cho phần lớn pixel.
* Alpha-only không đủ coverage để xóa logo độc lập.

## Experiment tiếp theo

Ba câu hỏi bắt buộc trước Experiment 3 đã có câu trả lời định lượng trong `RESEARCH.md` và `diagnostics/experiment2/report.json`.

Experiment 3 có thể nghiên cứu temporal reconstruction từ frame trước/sau, **chưa thêm optical flow** cho đến Experiment 4. Alpha/confidence maps của Experiment 2 nên được giữ làm nguồn ưu tiên trong hybrid sau này.

Experiment 3 chưa được triển khai.
