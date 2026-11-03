# NIH ChestX-ray14 — lớp ID/OOD
Chuỗi nhãn đúng như trong cột `Finding Labels` (Data_Entry_2017_v2020.csv). Chỉ dùng ảnh một nhãn.

## Quyết định
- ID (4): `No Finding`, `Infiltration`, `Effusion`, `Atelectasis`
- OOD (10): `Pneumothorax`, `Mass`, `Nodule`, `Cardiomegaly`, `Emphysema`, `Fibrosis`, `Pleural_Thickening`, `Consolidation`, `Edema`, `Pneumonia`
- Loại bỏ: `Hernia` (rất hiếm).
- Chọn prompt theo AUROC: OOD gộp cả 10 lớp OOD.
- Tách far-OOD {Pneumothorax, Mass, Nodule, Cardiomegaly, Emphysema, Fibrosis} và near-OOD {Consolidation, Edema, Pneumonia, Pleural_Thickening}: CHỈ để báo cáo, không dùng để chọn prompt hay tính go.

## Dự phòng (đăng ký trước)
- Chỉ kích hoạt khi Bước 0 DỪNG vì P_sel, P_cal hoặc P_test < 200 bệnh nhân.
- Khi đó: ID = `No Finding`, `Infiltration`; OOD = 10 lớp trên + `Effusion`, `Atelectasis`; Hernia vẫn loại. Agent không tự kích hoạt; Vinh xác nhận sau khi xem báo cáo Bước 0.
- Bước 0 in số phim một nhãn theo từng nhãn.

## Lý do
- ID là các nhãn phổ biến nhất trong NIH (Wang et al. 2017), để tập bệnh nhân ≥ 8 phim ID còn đủ ≥ 200/nhóm. Chưa kiểm trên ảnh một nhãn; xác minh ở Bước 0.
- OOD khác ngữ nghĩa với ID. Consolidation, Edema, Pneumonia gần Infiltration; Pleural_Thickening gần Effusion (suy luận) ⇒ báo riêng near-OOD.

## Phương án đã cân nhắc
- Đưa Hernia vào OOD: không chọn vì quá ít ảnh.
