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
