# E2FGVI-HQ CPU Research

## Phạm vi và trạng thái

Nhánh nghiên cứu: `research/e2fgvi-hq`.

Repo chính thức được clone cô lập tại `third_party/E2FGVI` ở commit
`709cbe319edc21b8a365a28e14cba595a93d62cf`. Không có file nào trong repo
upstream bị sửa và E2FGVI chưa được tích hợp vào CLI/pipeline chính.

Thiết lập lại clone và dependency nghiên cứu:

```powershell
git clone https://github.com/MCG-NKU/E2FGVI.git third_party/E2FGVI
git -C third_party/E2FGVI checkout 709cbe319edc21b8a365a28e14cba595a93d62cf
.\.venv\Scripts\python.exe -m pip install -r research/e2fgvi_hq/requirements.txt
```

Kết luận hiện tại: **E2FGVI-HQ chạy được hoàn toàn bằng CPU và cho chất lượng
đường nét tốt hơn các baseline hiện có, nhưng chưa đủ ổn định/tốc độ để chạy
toàn bộ video hoặc tích hợp vào pipeline chính.** Benchmark dừng ở đúng 2 giây.

## 1. Kiến trúc inference của repo

Luồng trong `test.py` và `model/e2fgvi_hq.py`:

1. Chia video thành các cửa sổ frame lân cận, mặc định tâm cách nhau 5 frame.
2. Thêm reference frame lấy theo bước 10 trên toàn đoạn.
3. Đưa vùng mask về 0 trong input của model.
4. SPyNet ước lượng optical flow hai chiều ở 1/4 độ phân giải.
5. Encoder trích feature; bidirectional second-order deformable propagation căn
   feature theo flow.
6. Tám Temporal Focal Transformer block tổng hợp thông tin temporal và reference.
7. Decoder sinh ảnh; các dự đoán cửa sổ chồng lấn được lấy trung bình.
8. Chỉ pixel trong mask được lấy từ prediction, pixel ngoài mask giữ nguyên.

E2FGVI-HQ hỗ trợ kích thước tùy ý bằng mirror padding. Crop 256×256 thực tế trở
thành 300×324 để feature map phù hợp cửa sổ transformer 60×108.

## 2. Checkpoint

- Tên upstream: `E2FGVI-HQ-CVPR22.pth`.
- Google Drive ID do README chính thức công bố:
  `10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3`.
- Kích thước: 164.535.938 byte.
- SHA-256:
  `afff989d41205598a79ce24630b9c83af4b0a06f45b137979a25937d94c121a5`.
- State dict: 244 tensor, gồm đủ 62 tensor của `update_spynet`.
- I3D checkpoint chỉ dùng cho đánh giá VFID, không cần cho inference.

Checkpoint lớn được ignore khỏi Git. Có thể tải và kiểm checksum bằng:

```powershell
.\.venv\Scripts\python.exe -m research.e2fgvi_hq.download_checkpoint
```

## 3. CPU-only và dependency Windows

PyTorch graph cốt lõi chạy được trên CPU. Vướng mắc không phải model mà là môi
trường upstream cũ:

- `environment.yml` khóa Linux, Python 3.7.13, PyTorch 1.5.1 CUDA 10.1,
  torchvision 0.6.1 CUDA và `mmcv-full==1.4.8`.
- `mmcv-full` chứa compiled modulated deformable convolution; không có wheel phù
  hợp tổ hợp Windows + Python 3.11 + PyTorch 2.7 CPU hiện tại.
- `test.py` còn kéo matplotlib/PyQt để hiển thị GUI và tự dilate mask 4 lần; cả hai
  đều không cần cho proof-of-concept này.

Giải pháp tối thiểu là shim `mmcv` chỉ trong process nghiên cứu:

- `mmcv.cnn.ConvModule` → `torch.nn.Conv2d` + ReLU, giữ nguyên key
  `conv.weight`/`conv.bias` để nạp checkpoint strict.
- `mmcv.ops.modulated_deform_conv2d` →
  `torchvision.ops.deform_conv2d`, đã kiểm tra backend CPU.
- `constant_init` → `torch.nn.init.constant_`.
- Bỏ lần tải SPyNet phụ khi constructor chạy; state dict HQ chính đã chứa đủ
  trọng số SPyNet và được nạp strict ngay sau đó.

Không sửa `test.py`, model code hay module nào của upstream. File nghiên cứu:

- `research/e2fgvi_hq/mmcv_cpu_shim.py`
- `research/e2fgvi_hq/smoke_test.py`
- `research/e2fgvi_hq/benchmark.py`
- `research/e2fgvi_hq/make_tight_diagnostics.py`
- `research/e2fgvi_hq/download_checkpoint.py`

Môi trường đã chạy: PyTorch `2.7.1+cpu`, torchvision `0.22.1+cpu`, không có CUDA.
License upstream là CC BY-NC 4.0; cần xem đây là hạn chế quan trọng nếu project
sau này dùng cho mục đích thương mại.

## 4. Thiết kế benchmark

- Video: `samples/ft-vid-23.mp4`, 1920×1080, 24 FPS.
- Đoạn: frame 108–155, đúng 48 frame / 2 giây; chứa DNA chuyển động và transition
  130→131.
- Context crop: `(x=1664, y=824, w=256, h=256)`; model không chạy full frame.
- Mask: mask Veo đã xác nhận, 186 pixel raw / 326 pixel sau dilation 1 px.
- Không dùng dilation 4 lần của upstream.
- Pixel ngoài 326 pixel mask: sai khác cực đại trước encode bằng 0.
- Inference parameters giữ flow chính thức: neighbor stride 5, reference step 10,
  10 cửa sổ chồng lấn.
- CPU threads: 4.

Smoke test 3 frame thật đạt 3,175 giây tổng, 0,945 FPS và peak RSS 921 MB trước
khi chạy benchmark 48 frame.

Lệnh benchmark:

```powershell
.\.venv\Scripts\python.exe -m research.e2fgvi_hq.benchmark `
  --start-frame 108 --frames 48 --crop-size 256 --threads 4
```

## 5. Kết quả định lượng

### Tài nguyên E2FGVI-HQ

| Chỉ số | Kết quả |
|---|---:|
| Tổng inference | 181,681 giây |
| Thời gian / output frame | 3,785 giây |
| Throughput | 0,264 FPS |
| Thời gian / cửa sổ, median | 19,269 giây |
| Peak RSS | 2.601,5 MB |
| Tham số model | 41.117.887 |
| CUDA build / available | `None` / `False` |

So với baseline CPU đã đo trước đó trên cùng máy: TELEA khoảng 0,000768
giây/frame, Alpha-only 0,000865 giây/frame, LaMa 256 khoảng 0,886 giây/frame.
E2FGVI chậm hơn LaMa khoảng 4,27 lần theo output frame vì mỗi frame tham gia nhiều
cửa sổ temporal chồng lấn.

### Proxy chất lượng trên 48 frame

`logo likeness` thấp hơn thường nghĩa là ít giống lớp chữ trắng hơn, nhưng không
phải ground truth. Temporal MAD thấp không luôn tốt: TELEA hạ MAD bằng cách làm
mờ/đứt chi tiết chuyển động.

| Phương pháp | Mean logo likeness ↓ | Mean mask temporal MAD | MAD 130→131 |
|---|---:|---:|---:|
| Original có watermark | 0,573617 | 20,834095 | 29,319018 |
| TELEA | 0,195206 | 15,827242 | 31,579755 |
| Alpha-only | 0,553718 | 20,677849 | 29,730061 |
| LaMa | 0,194727 | 26,802767 | 36,782209 |
| **E2FGVI-HQ** | **0,227565** | **22,426511** | **30,723926** |

E2FGVI có worst transition tại 132→133: MAD 57,261 sau encode, trong khi original
có watermark là 38,972 ở cùng transition. Đây là dấu hiệu thay đổi độ sáng/texture
cục bộ do prediction, dù phần lớn biến đổi quan sát được cũng đến từ đường DNA
đang chạy rất nhanh qua mask.

Số đo đầy đủ nằm tại
`research/e2fgvi_hq/results/benchmark_report.json`.

## 6. Đánh giá trực quan

### Logo còn sót

- TELEA và LaMa xóa hình chữ rõ nhất theo proxy nhưng thay bằng vùng tối dễ thấy.
- Alpha-only giữ gần như toàn bộ hình chữ vì chỉ một phần pixel đủ confidence.
- E2FGVI không còn ba ký tự Veo dễ nhận ra. Một ít sáng/tối trong đúng stroke mask
  là nội dung dự đoán, không tạo lại hình chữ hoàn chỉnh.

### DNA, đường sáng và texture

- Frame 128–134 là test quan trọng: đường cyan chạy ngang qua watermark.
- TELEA tạo smear tối và làm đứt hoặc lệch đường.
- LaMa tạo blotch tối, có lúc sinh cạnh chéo giả và thay đổi mạnh giữa frame.
- Alpha-only giữ nét quan sát được nhưng cũng giữ watermark.
- E2FGVI nối đường ngang tự nhiên nhất, giữ hướng và màu gần vùng hai bên. Ở frame
  130→131, MAD gần original hơn LaMa và TELEA.

### Ghosting và flicker

- Không thấy double-edge/ghost kéo dài rõ trong chuỗi phóng đại frame 127–135.
- Có brightness/texture wobble ngắn ở 132→133 và một số prediction chuyển từ tối
  sang nét sáng nhanh hơn tín hiệu quan sát được. Đây là artifact còn lại quan
  trọng nhất.
- Không có patch đen phẳng. Vùng tối ở một số frame bám theo nền nhưng đôi lúc
  lệch sáng so với texture xung quanh.
- Đánh giá chỉ dựa trên video có watermark, không có clean ground truth; vì vậy
  không thể tuyên bố đường hallucinate là background thật tuyệt đối.

Diagnostic local (không commit vì là artifact lớn):

- `research/e2fgvi_hq/outputs/five_method_comparison.mp4`
- `research/e2fgvi_hq/outputs/e2fgvi_hq_crop.mp4`
- `research/e2fgvi_hq/outputs/comparison_contact_sheet.png`
- `research/e2fgvi_hq/outputs/transition_130_131.png`
- `research/e2fgvi_hq/outputs/tight_roi/`

## 7. Quyết định

CPU-only **khả thi về kỹ thuật**; CUDA/mmcv-full không còn là blocker và không cần
patch sâu repo. E2FGVI-HQ là baseline đầu tiên trong nhánh AI giữ được đường sáng
qua logo thuyết phục hơn TELEA và LaMa.

Tuy nhiên chưa chạy full video và chưa tích hợp pipeline chính vì:

1. throughput chỉ 0,264 FPS;
2. peak RAM khoảng 2,6 GB cho riêng crop 256;
3. spike 132→133 cho thấy vẫn có flicker/brightness wobble cục bộ;
4. license CC BY-NC 4.0 không phù hợp để mặc định đưa vào sản phẩm thương mại.

Nếu tiếp tục nhánh này, thí nghiệm kế tiếp nên giữ nguyên model/checkpoint nhưng
đo ba biến riêng: crop 192 so với 256, số reference frame hữu hạn, và cách blend
các cửa sổ chồng lấn có trọng số theo khoảng cách tới tâm. Chỉ chạy full 8 giây
sau khi phương án giảm được spike temporal mà không làm đứt đường DNA.

---

## 8. Experiment tối ưu E2FGVI-HQ CPU

### Mục tiêu và nguyên tắc kiểm soát biến

Experiment giữ nguyên checkpoint, model, frame 108–155, mask 326 pixel, CPU 4
threads, neighbor stride 5 và toàn bộ metric của proof-of-concept. Không sửa
upstream, không tích hợp pipeline chính và không chạy video 8 giây.

Ba nhóm biến được chạy độc lập trước khi kết hợp:

1. crop 192/224/256, giữ reference step 10 và `legacy_average`;
2. reference step 10/20/30, giữ crop 256 và `legacy_average`;
3. `legacy_average`/`center_weighted`, giữ crop 256 và reference step 10.

`legacy_average` là đúng hành vi baseline hiện có: prediction chồng lấn được trộn
tuần tự 50/50. Khi có ba contribution, đây không phải mean số học tuyệt đối.
`center_weighted` dùng trọng số tuyến tính 6 ở frame tâm xuống 1 ở biên ±5.

Ngoài metric cũ, experiment ghi riêng MAD và signed mean-luma delta tại 130→131,
132→133, cùng mean Laplacian energy trên raw logo mask để theo dõi texture.

### 8.1 Crop/context độc lập

| Crop | Padding thật | giây/frame | Peak RSS MB | Logo likeness | Mean MAD | MAD 130→131 | MAD 132→133 | Luma Δ 132→133 | Laplacian |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 192 | 240×216 | 3,375 | 2.110,6 | 0,226101 | 22,209698 | 31,242331 | 54,981595 | -15,668716 | 68,675403 |
| 224 | 240×324 | 5,943 | 2.696,7 | 0,228899 | 22,622243 | 30,533742 | 53,837423 | -17,475464 | 68,010081 |
| 256 control repeat | 300×324 | 5,798 | 3.143,6 | 0,227565 | 22,426511 | 30,723926 | 57,156442 | -19,641098 | 71,995296 |

Kết luận crop:

- 224 không phải điểm giữa hiệu quả: width 224 bị pad lên 324, bằng crop 256, nên
  không giảm đủ tensor transformer và không thắng performance.
- 192 giảm padding xuống 240×216 và peak RAM khoảng 1 GB so với control repeat.
- Inspection frame 128–136 cho thấy 192 vẫn nối đúng hướng đường DNA/cyan qua
  watermark, không thêm ghost/double edge rõ. Texture trong mask mềm nhẹ hơn;
  Laplacian giảm 4,6% so với 256.
- 192 được chọn cho cấu hình kết hợp vì giảm MAD 132→133 và luma wobble, đồng thời
  không làm DNA continuity kém đi rõ rệt.

### 8.2 Temporal reference độc lập

| Reference step | Reference ngoài neighbor/cửa sổ | giây/frame đo được | Peak RSS MB | Mean MAD | MAD 130→131 | MAD 132→133 | Luma Δ 132→133 | Laplacian |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 control repeat | 3–4 | 5,798 | 3.143,6 | 22,426511 | 30,723926 | 57,156442 | -19,641098 | 71,995296 |
| 20 | 1–3 | 6,535 | 2.836,3 | 22,433690 | 30,779141 | 56,638037 | -20,208595 | 71,250224 |
| 30 | 1–2 | 7,689 | 2.281,2 | 22,481203 | 30,828221 | 56,975460 | -21,435577 | 71,650313 |

Step 20/30 giảm số reference và step 30 giảm RAM, nhưng không cải thiện DNA nhìn
thấy, không giảm spike có ý nghĩa và làm luma delta xấu hơn. Wall-clock cũng không
chứng minh được tăng tốc vì 6–11 neighbor frame và feature propagation vẫn chiếm
phần lớn chi phí. Cấu hình thắng của nhóm này vẫn là step 10.

### 8.3 Overlap aggregation độc lập

| Aggregation | Mean MAD | MAD 130→131 | MAD 132→133 | Luma Δ 132→133 | Laplacian |
|---|---:|---:|---:|---:|---:|
| `legacy_average` | 22,426511 | 30,723926 | 57,156442 | -19,641098 | 71,995296 |
| `center_weighted` | 22,578319 | 30,871166 | 58,536810 | -19,794479 | 72,717294 |

Center weighting giữ texture sắc hơn nhẹ nhưng làm cả mean MAD và spike 132→133
xấu hơn. Inspection cho thấy độ sáng ở phần đường chạy qua overlap thay đổi rõ hơn,
không có lợi về ghosting. Giữ `legacy_average`.

### 8.4 Lưu ý về repeatability của thời gian CPU

Các run liên tiếp có thermal/system-load variability lớn. Cùng cấu hình 256 cho
output deterministic giống hệt từng metric nhưng tổng inference thay đổi từ
181,68 giây ở proof-of-concept tới 278,29 giây ở control repeat. Các sweep về sau
cũng chậm hơn khi CPU chịu tải liên tục. Vì vậy:

- số chất lượng có thể so sánh trực tiếp vì output deterministic;
- thời gian từng run được giữ nguyên, không loại bỏ số xấu;
- quyết định performance cuối so baseline gốc với một rerun final riêng, đồng thời
  dùng padded shape/peak RAM để kiểm tra xu hướng.

### 8.5 Cấu hình thắng và rerun final 48 frame

```text
crop_size       = 192
internal_pad    = 240 x 216
neighbor_stride = 5
reference_step  = 10
aggregation     = legacy_average
threads         = 4
```

| Chỉ số | Baseline 256 gốc | Final 192 | Thay đổi |
|---|---:|---:|---:|
| Tổng inference | 181,681 s | 119,370 s | nhanh hơn 34,3% |
| Giây/output frame | 3,785 | 2,487 | nhanh hơn 34,3% |
| Throughput | 0,264 FPS | 0,402 FPS | tăng 52,2% |
| Peak RSS | 2.601,5 MB | 2.114,6 MB | giảm 18,7% |
| Logo likeness | 0,227565 | 0,226101 | tốt hơn nhẹ |
| Mean temporal MAD | 22,426511 | 22,209698 | giảm 1,0% |
| MAD 130→131 | 30,723926 | 31,242331 | tăng 1,7% |
| MAD 132→133 | 57,156442 | 54,981595 | giảm 3,8% |
| Luma Δ 132→133 | -19,641098 | -15,668716 | giảm độ lớn 20,2% |
| Laplacian | 71,995296 | 68,675403 | giảm 4,6% |

Đánh giá final:

- đường DNA/cyan tại 128–134 vẫn liên tục và đúng hướng; không thấy mất context rõ
  khi giảm từ 256 xuống 192;
- không còn logo dễ nhận ra, không có patch đen phẳng;
- không thấy ghost/double edge kéo dài;
- brightness/texture wobble 132→133 giảm nhưng chưa biến mất;
- trade-off chính là texture trong 326 pixel mask mềm hơn nhẹ và transition
  130→131 tăng MAD nhỏ.

Mục tiêu experiment đạt: cấu hình final nhanh hơn baseline, dùng ít RAM hơn và giảm
spike 132→133 mà không làm chất lượng DNA kém đi rõ rệt. Dù vậy vẫn chưa chạy full
8 giây: 0,402 FPS còn chậm và temporal artifact chưa đủ thấp để tích hợp pipeline.

Report có thể truy vết:

- `research/e2fgvi_hq/results/optimization/optimization_summary.json`
- `research/e2fgvi_hq/results/optimization/final_best.json`
- các report độc lập `crop_*.json`, `ref_step_*.json`, `weighted_256.json`.

---

## 9. Full-video Validation — 192 frame / 8 giây

### Phạm vi

Validation dùng nguyên cấu hình thắng, không tuning thêm:

```text
crop_size       = 192
internal_pad    = 240 x 216
neighbor_stride = 5
reference_step  = 10
aggregation     = legacy_average
threads         = 4
```

Model/checkpoint, mask 326 pixel và code upstream không đổi. AI chỉ chạy trên ROI
192×192; sau inference, crop được ghép streaming vào frame gốc 1920×1080. Không
thêm smoothing, flow ngoài model hoặc phương pháp phục hồi mới.

### Output và media integrity

| Thuộc tính | Input | Output |
|---|---:|---:|
| Resolution | 1920×1080 | 1920×1080 |
| FPS | 24 | 24 |
| Frame count | 192 | 192 |
| Duration | 8,0 s | 8,0 s |
| Video codec | H.264 | H.264 |
| Audio | AAC | AAC, stream copy |

AAC được trích lại dưới dạng ADTS từ input/output và có cùng SHA-256:

```text
9b4130e48c041645eff681b635849880a549a78b414eea049430f479d2b37b38
```

### Hiệu năng full video

| Chỉ số | Kết quả |
|---|---:|
| Tổng inference | 1.271,366 giây |
| Inference/frame | 6,621696 giây |
| Throughput | 0,151019 FPS |
| Encode + mux | 6,319 giây |
| Tổng validation | 1.287,655 giây / 21,46 phút |
| Peak RSS | 4.216,637 MB |

Full run chậm hơn ngoại suy từ đoạn 48 frame vì mỗi cửa sổ tham chiếu toàn bộ video:
25–30 frame input/cửa sổ thay vì khoảng 10–15. Có 39 cửa sổ temporal. Runtime từng
cửa sổ dao động 17,9–60,8 giây do tải/thermal CPU.

### Temporal metrics toàn video

| Phương pháp | Mean MAD | Median | P75 | Max |
|---|---:|---:|---:|---:|
| Original có watermark | 15,584685 | 15,328221 | 20,699387 | 48,858896 |
| TELEA | 11,075194 | 9,027607 | 16,489264 | 31,579755 |
| LaMa | 19,373414 | 18,018405 | 26,139571 | 54,546012 |
| **E2FGVI-HQ** | **14,675810** | **12,475460** | **20,753067** | **57,748466** |

TELEA có MAD thấp vì làm mềm/đứt chi tiết, không đồng nghĩa chất lượng tốt hơn.
E2FGVI có mean MAD thấp hơn original và LaMa, nhưng max cao nhất và tập trung vào
một cụm chuyển động khó.

Hai transition trọng điểm:

| Transition | Original MAD | TELEA | LaMa | E2FGVI-HQ |
|---|---:|---:|---:|---:|
| 130→131 | 29,319018 | 31,579755 | 36,782209 | 32,021472 |
| 132→133 | 38,972393 | 17,432515 | 48,297546 | 57,748466 |

### Top 10 transition E2FGVI mạnh nhất

| Hạng | Transition | MAD | Mean-luma delta |
|---:|---:|---:|---:|
| 1 | 132→133 | 57,748466 | -15,920250 |
| 2 | 131→132 | 45,714724 | -0,960121 |
| 3 | 133→134 | 41,555215 | -19,598156 |
| 4 | 122→123 | 37,668712 | 2,907974 |
| 5 | 121→122 | 35,404908 | 3,754601 |
| 6 | 43→44 | 34,871166 | -6,386505 |
| 7 | 32→33 | 34,297546 | 0,475464 |
| 8 | 123→124 | 33,334356 | -5,561348 |
| 9 | 34→35 | 32,414110 | 5,579750 |
| 10 | 149→150 | 32,383436 | 5,917175 |

Ở 132→133, original có MAD 38,972393 và luma delta -9,702454; E2FGVI tăng lên
57,748466 và -15,920250. Đây là bằng chứng định lượng rằng wobble còn nghiêm trọng
cục bộ, dù phần lớn video ổn định hơn LaMa.

### Logo, DNA, ghosting, seam và patch tối

- Mean logo likeness: original 0,582052; TELEA 0,224637; LaMa 0,212697;
  E2FGVI 0,241284. Inspection không thấy ba ký tự Veo còn nhận ra rõ ở output.
- E2FGVI giữ DNA/cyan edge liên tục tốt nhất; TELEA smear/đứt nét, LaMa sinh blotch
  tối và texture giả rõ hơn.
- Không thấy ghost/double-edge kéo dài trong top-10 transition.
- Max absolute change ngoài mask và tại boundary crop trước encode đều bằng 0;
  không có seam ROI do crop chỉ là context, không phải vùng replace toàn phần.
- Dark-patch proxy thấp nhất tại frame 136 (-40,108723), sau đó 12–15 và 101–103.
  Phần lớn phản ánh background thật đang tối; không thấy patch đen phẳng. Tuy nhiên
  sequence 100–103 còn imprint tối/texture mềm nhẹ tại mask.
- Sequence 127–136 giữ đường DNA hợp lý nhưng brightness/texture đổi mạnh tại
  132–134, khớp spike temporal.

### Kết luận Full-video Validation

**Cấu hình hiện tại chưa đủ ổn định để đạt Definition of Done hoặc tích hợp
pipeline chính.** Lý do quyết định là spike 132–134 cao hơn chuyển động quan sát
được trong source, cùng luma wobble và imprint texture cục bộ. Đây không phải lỗi
seam, audio, metadata hay logo còn sót.

E2FGVI-HQ vẫn là phương pháp nghiên cứu có chất lượng phục hồi đường/DNA tốt nhất
đã thử: ít ghost hơn LaMa, ít smear hơn TELEA và mean temporal MAD toàn video hợp
lý. Nhưng chất lượng trung bình tốt không bù được failure cục bộ rõ tại transition
khó. Không tự động thêm smoothing hoặc đổi model trong milestone này.

Artifacts:

- report: `research/e2fgvi_hq/results/full_validation_report.json`;
- full video: `research/e2fgvi_hq/outputs/full_validation/ft-vid-23_e2fgvi_hq_cpu_full.mp4`;
- diagnostics: `top_transitions/`, `top_transitions_tight/`, `dark_patch_frames/`,
  `sequences/` trong thư mục output full validation.
