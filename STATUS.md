# Trạng thái Project

## Phương pháp tốt nhất hiện tại

Chưa có phương pháp nào đạt Definition of Done.

* TELEA xóa logo nhưng blur/smear, đứt DNA và flicker.
* Alpha-only giữ dữ liệu thật nhưng chỉ xử lý 25/326 pixel mask.
* Direct temporal và Optical Flow có coverage quá thấp.
* LaMa CPU xóa chữ tốt hơn TELEA một chút nhưng không phục hồi đúng DNA và có flicker cao nhất.

## Milestone hiện tại

**AI Experiment 1 — LaMa CPU đã hoàn tất trên toàn bộ benchmark 192 frame.**

Implementation dùng model TorchScript `big-lama.pt` tương đương IOPaint:

```text
torch 2.7.1+cpu
CUDA build: None
CUDA available: False
device: CPU
```

Không dùng Optical Flow, model video, full-frame AI inference, blur hoặc flat patch.

## Crop và mask

* Đã thử context 192×192 và 256×256 trên 12 frame dễ/khó.
* Video đầy đủ dùng crop 256×256 tại góc dưới phải.
* Mask giữ nguyên 326 pixel từ Experiment 1.
* LaMa chỉ sửa pixel trong mask; max absolute change ngoài mask trước encode bằng 0.
* 192 và 256 cho chất lượng gần như tương đương; 192 nhanh hơn khoảng 27–37% tùy lượt đo.

## Kết quả chất lượng

| Metric | TELEA | Alpha-only | LaMa 256 |
|---|---:|---:|---:|
| Logo-likeness | 0,233025 | 0,567375 | **0,222844** |
| Temporal MAD | 10,884512 | 15,791845 | **19,735217** |
| Transition 130→131 MAD | 30,993865 | 29,631902 | **36,579755** |
| Raw-mask Laplacian energy | 35,56 | 504,13 | 143,97 |

LaMa xóa hình dạng logo khá sạch và giữ nhiều chi tiết tần số cao hơn TELEA. Tuy nhiên:

* DNA/đường sáng vẫn bị đứt;
* model sinh mảng tối và texture không đúng;
* patch thay đổi hình dạng giữa các frame;
* transition 130→131 flicker rõ;
* worst transition là 43→44 với MAD 55,748466.

Kết luận: **LaMa per-frame không đủ chất lượng để tiếp tục như phương pháp độc lập.**

## Hiệu năng

* TELEA: 0,768 ms/frame.
* Alpha: 0,865 ms/frame.
* LaMa 256 CPU: 0,886 s/frame, khoảng 1,128 processing FPS.
* Tổng benchmark: 202,02 giây.

## Output

```text
outputs/ai_experiment1/ft-vid-23_lama_cpu.mp4
diagnostics/ai_experiment1/
```

Video giữ 1920×1080, 24 FPS, 192 frame, 8 giây và audio AAC gốc. Audio SHA-256 trùng input. Model local trong `models/` đã được gitignore và xác minh MD5 upstream.

## Bước tiếp theo

Không tự động chuyển sang Optical Flow hoặc model video trong milestone này. LaMa được giữ làm baseline ảnh đơn. Nếu tiếp tục hướng AI, experiment mới phải có temporal conditioning/consistency và phải chứng minh giảm flicker tại 130→131 mà không phá DNA.
