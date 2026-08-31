# Trạng thái Project

## Phương pháp tốt nhất hiện tại

**E2FGVI-HQ CPU với crop 192×192** hiện cho chất lượng phục hồi đường DNA tốt nhất
trong các phương pháp đã thử, nhưng chưa đạt Definition of Done vì còn flicker và
brightness/texture wobble cục bộ tại frame 132–134.

Các baseline:

- TELEA xóa logo nhưng blur/smear và làm đứt DNA.
- Alpha-only giữ dữ liệu thật nhưng chỉ xử lý một phần rất nhỏ mask.
- Direct temporal và Optical Flow cổ điển có coverage quá thấp.
- LaMa xóa chữ nhưng sinh blotch tối, texture giả và flicker cao.
- E2FGVI-HQ giữ DNA/đường sáng tốt nhất, không còn logo dễ nhận ra và không có seam,
  nhưng worst transition vẫn cao hơn chuyển động quan sát được trong source.

## Milestone hiện tại

**Full-video Validation E2FGVI-HQ CPU đã hoàn tất trên toàn bộ 192 frame / 8 giây.**

```text
crop_size       = 192
internal_pad    = 240 x 216
neighbor_stride = 5
reference_step  = 10
aggregation     = legacy_average
threads         = 4
device          = CPU
CUDA available  = False
```

Không sửa upstream, không tích hợp pipeline chính và không thêm temporal smoothing
hay phương pháp mới.

## Output integrity

- Resolution: 1920×1080.
- FPS: 24.
- Frame count: 192.
- Duration: 8,0 giây.
- Video: H.264.
- Audio: AAC stream copy; SHA-256 bitstream trùng input.
- Max thay đổi ngoài mask trước encode: 0.
- Max thay đổi tại crop boundary trước encode: 0.

## Hiệu năng

- Tổng inference: 1.271,366 giây.
- Inference/frame: 6,621696 giây.
- Throughput: 0,151019 FPS.
- Encode/mux: 6,319 giây.
- Tổng validation: 1.287,655 giây, khoảng 21,46 phút.
- Peak RSS: 4.216,637 MB.

## Chất lượng toàn video

| Phương pháp | Mean temporal MAD | Worst MAD |
|---|---:|---:|
| Original | 15,584685 | 48,858896 |
| TELEA | 11,075194 | 31,579755 |
| LaMa | 19,373414 | 54,546012 |
| E2FGVI-HQ | 14,675810 | 57,748466 |

Top failure E2FGVI:

1. 132→133: MAD 57,748466, luma delta -15,920250.
2. 131→132: MAD 45,714724.
3. 133→134: MAD 41,555215, luma delta -19,598156.

Đánh giá trực quan:

- logo không còn nhận ra rõ;
- DNA/cyan edge liên tục tốt hơn TELEA và LaMa;
- không có seam ROI hoặc patch đen phẳng;
- không thấy ghost/double-edge kéo dài;
- còn wobble mạnh tại 132–134;
- còn imprint tối/texture mềm nhẹ tại một số frame như 100–103.

## Quyết định

**Chưa tích hợp E2FGVI-HQ vào pipeline chính và chưa coi là hoàn thành.**

Full validation chứng minh model chạy CPU end-to-end và giữ media đúng, nhưng failure
cục bộ 132–134 còn quá rõ. Bước nghiên cứu tiếp theo phải tập trung vào cách giảm
temporal outlier mà không làm mềm hoặc phá DNA; không được đánh đổi bằng blur.

## Artifacts

```text
research/e2fgvi_hq/results/full_validation_report.json
research/e2fgvi_hq/outputs/full_validation/ft-vid-23_e2fgvi_hq_cpu_full.mp4
research/e2fgvi_hq/outputs/full_validation/top_transitions/
research/e2fgvi_hq/outputs/full_validation/top_transitions_tight/
research/e2fgvi_hq/outputs/full_validation/dark_patch_frames/
research/e2fgvi_hq/outputs/full_validation/sequences/
```
