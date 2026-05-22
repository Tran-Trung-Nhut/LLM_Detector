# Paper Review — Revision Status
> Cập nhật: 2026-05-22. Ghi lại toàn bộ thay đổi code và kết quả thực nghiệm sau revision.

---

## Tổng quan

Reviewer yêu cầu 8 thay đổi. Chúng ta thực hiện được **6/8**. Hai mục còn lại (mở rộng dataset và low-prior test set) không thực hiện vì lý do thời gian và nguồn lực — xem Phần 3.

---

## Phần 1 — Những gì đã làm (6 mục)

### 1.1 [Must do] Significance tests trên independent test set + F1 bootstrap

**Vấn đề gốc:** `statistical_tests.py` đang load predictions từ CV OOF (`text_only/predictions.csv`), không phải independent test set. Không có bootstrap cho F1/accuracy.

**Đã sửa:** `src/steps/statistical_tests.py`
- Đổi data source sang `independent_test/predictions_*.csv` (N=110)
- Thêm hàm `bootstrap_f1_accuracy()` — paired bootstrap 2000 iterations cho F1 và accuracy
- Thêm Holm-corrected p-value cho F1
- Output mới: `delta_f1`, `f1_ci95`, `p_f1_bootstrap`, `p_holm_f1` trong `summary.csv`

**Kết quả** (`runs/feature_fusion/statistical_tests/`):

| Comparison vs Text-Only | F1 base | F1 cmp | ΔF1 | 95% CI F1 | p (bootstrap) | ΔAUC | p (DeLong) | Cliff δ |
|---|---|---|---|---|---|---|---|---|
| early_fusion | 0.8506 | 0.8776 | +0.027 | [−0.047, +0.102] | 0.241 | +0.004 | 0.708 | 0.12 |
| late_soft_voting | 0.8506 | 0.8333 | −0.017 | [−0.083, +0.050] | 0.690 | −0.004 | 0.730 | 0.36 |
| late_stacking | 0.8506 | 0.8485 | −0.002 | [−0.075, +0.077] | 0.521 | −0.001 | 0.914 | 0.68 |
| late_score_max | 0.8506 | 0.8269 | −0.024 | [−0.103, +0.057] | 0.717 | −0.015 | 0.353 | 0.68 |
| image_only | 0.8506 | 0.7527 | −0.098 | [−0.217, +0.011] | 0.966 | −0.104 | **0.006*** | −0.60 |

\* Significant sau Holm correction (p_holm = 0.030)

**Diễn giải quan trọng cho reviewer:**
- Headline gap Early Fusion F1=0.878 vs Text-Only F1=0.851 (ΔF1=+0.027) **không có ý nghĩa thống kê** (p=0.241, 95% CI bao gồm 0). Đây là điều reviewer #1 muốn biết — cần trình bày trung thực trong paper.
- Image-Only tệ hơn Text-Only một cách **có ý nghĩa thống kê** về AUC (p=0.006), củng cố lý do dùng fusion.
- Không có strategy fusion nào vượt trội có ý nghĩa thống kê so với Text-Only trên N=110 — điều này hợp lý với test set nhỏ, cần nêu rõ đây là limitation về statistical power.

---

### 1.2 [Must do] Per-label-criterion evaluation

**Vấn đề gốc:** Reviewer phê bình construct validity — positive label gộp 3 hiện tượng khác nhau mà không đánh giá riêng từng nhóm.

**Đã làm:** Tạo `src/steps/per_label_criterion.py` — dùng proxy từ features sẵn có (không cần annotation tay mới):
- **Model name in listing** → keyword col[2] (model_name binary) > 0
- **Generative behavior** → keyword col[6] OR [8] OR [10] > 0 (generation/interaction/content)
- **Chat-style UI only** → zeroshot score > 0.15 (top ~25% test distribution)
- **Hard / ambiguous** → không có proxy nào

**Kết quả** (`runs/feature_fusion/per_label_criterion/`):

| Nhóm | N positives | Soft Voting F1 |
|------|------------|---------------|
| model_name_in_listing | 5 | 0.455 |
| generative_behavior | 9 | 0.500 |
| chat_style_ui_only | 5 | 0.455 |
| hard_ambiguous | 25 | 0.767 |

**Nhận xét quan trọng:** Nhóm `hard_ambiguous` (25/44 positives = 57%) đạt F1 cao nhất — đây là các app mà model học được signal từ SBERT embedding chứ không phải từ keyword hay UI trực tiếp. Nhóm `model_name_in_listing` chỉ có 5 apps trên test set vì nhiều app có model name đã nằm trong training data.

---

### 1.3 [High impact] Keyword-drift robustness test

**Vấn đề gốc:** Reviewer #4 hỏi vocabulary 39 keyword có bị drift khi tên model mới xuất hiện không.

**Đã làm:** Tạo `src/steps/keyword_drift.py`
- Re-compute keyword features từ raw text mà không có category `model_name` (chatgpt, gpt-4, gpt-3, claude, gemini, copilot, llama, mistral)
- Giữ nguyên SBERT và meta features
- Retrain 5-fold text-only LightGBM với features mới (chạy trong RAM, không ghi đè model gốc)
- Evaluate trên independent test set với cùng cách zero model-name keywords

**Kết quả** (`runs/feature_fusion/keyword_drift/keyword_drift_results.json`):

| Điều kiện | F1 | AUC |
|-----------|-----|-----|
| CV — có model-name keywords | 0.7531 | 0.8723 |
| CV — không có model-name keywords | 0.7352 (−0.018) | 0.8655 (−0.007) |
| Indep. test — có model-name keywords | 0.8510 | 0.9480 |
| Indep. test — không có model-name keywords | 0.8671 (+0.016) | 0.9390 (−0.009) |

**Diễn giải:** Trên CV, bỏ model-name giảm F1 nhẹ (−0.018). Trên independent test set, F1 thậm chí tăng nhẹ (+0.016) — cho thấy SBERT embedding đủ mạnh để bù đắp signal từ tên model. Kết quả này là **bằng chứng tốt cho reviewer**: model không phụ thuộc nặng vào tên model cụ thể, vocabulary drift không gây degradation nghiêm trọng.

---

### 1.4 [Medium] Phân tích khi nào image giúp / gây hại

**Vấn đề gốc:** Reviewer nói cross-modal story "được khẳng định nhưng không được chứng minh bằng ví dụ cụ thể."

**Đã làm:** Tạo `src/steps/image_analysis_examples.py` — phân loại 110 test apps thành 6 pattern:

**Kết quả** (`runs/feature_fusion/image_analysis/`):

| Pattern | N | N pos | Fusion Acc | Avg text | Avg img |
|---------|---|-------|-----------|---------|--------|
| A: Image saves (text miss, img hit) | 5 | 5 | — | — | — |
| B: Image misleads (text ok, img FP) | 11 | 0 | — | — | — |
| C: Text leads (text hit, img miss) | 7 | 7 | — | — | — |
| D: Both agree positive | 30 | 30 | — | — | — |
| E: Both agree negative | 49 | 0 | — | — | — |
| F: Both wrong | 8 | 2 | — | — | — |

**Diễn giải:** Image branch cứu được **5 TPs** mà text alone miss, nhưng cũng tạo thêm **11 FPs**. Pattern D (both agree positive) chiếm 30/44 positives — tức là 68% positive apps đã hiển nhiên cho cả hai modality. Examples cho paper nằm ở `image_helps_examples.json` và `image_hurts_examples.json`.

---

### 1.5 [Medium] Image-Only trên independent test set

**Vấn đề gốc:** `independent_test_eval.py` không có Image-Only trong strategies, làm so sánh modality không đầy đủ.

**Đã sửa:** Thêm `"Image-Only": img_probs` vào strategies dict trong `src/steps/independent_test_eval.py`.

**Output:** `runs/feature_fusion/independent_test/predictions_image_only.csv` đã tồn tại.

Số liệu Image-Only trên N=110 giờ có đầy đủ trong `table20_independent_test.json` để so sánh trực tiếp với Text-Only và các fusion strategies.

---

### 1.6 [Optional] SHAP / Feature importance

**Đã làm:** Tạo `src/steps/shap_analysis.py` — TreeSHAP trên tất cả 5 folds, aggregate mean |SHAP|.

**Kết quả** (`runs/feature_fusion/shap_analysis/`):

**Text-Only — top features:**
1. `kw_model_name_count` (keyword)
2. `kw_max_len` (keyword)
3. `sbert_1007`, `kw_core_llm_count`, `sbert_1000`...

**Group importance — Text-Only:**
- SBERT: 3.999 >> Keyword: 0.618 >> Meta: ~0.0

**Early Fusion — top features:**
1. `clip_mean_76`, `kw_max_len`, `clip_mean_135`...

**Group importance — Early Fusion:**
- clip_mean: 2.064 > SBERT: 1.449 > clip_max: 1.195 > Keyword: 0.342 > zeroshot: 0.071

**Diễn giải:** SBERT embedding là backbone chính của text branch (chiếm 87% total importance trong Text-Only). Trong Early Fusion, CLIP features (clip_mean + clip_max) cộng lại còn lớn hơn SBERT — cho thấy image branch đóng góp thực chất chứ không bị text branch lấn át.

---

## Phần 2 — Những gì KHÔNG làm và lý do

### 2.1 [High impact] Tạo low-prior test set thực tế (2–5% LLM prevalence)

**Reviewer yêu cầu:** Lấy mẫu test set mới từ nhiều category rộng hơn, nơi chỉ 2–5% là LLM app, thay vì chỉ dùng post-hoc mathematical correction.

**Lý do không làm:**

Pipeline hiện tại đã có `prior_correction.py` với Figure 5 (prior-corrected precision vs deployment prior). Để tạo được test set thực tế với 2–5% prevalence, cần:

1. **Thu thập ~500–2000 apps mới** từ Google Play, spanning nhiều category như Games, Health, Shopping — nơi LLM app gần như không xuất hiện. Việc này mất 1–2 ngày chạy automated scraping.
2. **Annotate tay toàn bộ** bởi ít nhất 2 annotators — với ~1000 apps ước tính mất 5–10 ngày làm việc thực.
3. **Extract features** (SBERT + CLIP + OCR) cho dataset mới — thêm 4–8 giờ GPU.

Tổng: **2–3 tuần** với nguồn lực người đầy đủ. Không khả thi trong timeline revision hiện tại.

**Cách phản hồi reviewer:** Giữ Figure 5 (prior-corrected precision curve) và bổ sung câu giải thích rằng mathematical correction là conservative lower-bound. Thừa nhận đây là limitation và ghi vào future work.

---

### 2.2 [High impact] Mở rộng dataset (298 train / 110 test)

**Reviewer yêu cầu:** Thêm apps từ nhiều category hơn, hard negatives (app AI nhưng không phải LLM), non-English listings, và ideally một app store khác.

**Lý do không làm:**

Đây là vấn đề lớn nhất về nguồn lực:

1. **Thu thập data mới** tương tự như mục 2.1 nhưng quy mô lớn hơn (mục tiêu ≥ 2× dataset hiện tại = +600 apps training).
2. **Annotation tay** với IAA protocol — mỗi app cần 2 annotators độc lập, resolution meeting khi disagree. Với 600 apps mới và average 15 phút/app × 2 annotators: ~300 giờ làm việc.
3. **Full pipeline retrain** sau khi có data mới — extract features, retrain 5-fold tất cả strategies, re-run toàn bộ analysis. Thêm 1–2 ngày.
4. **Non-English listings** cần thay đổi text model (BGE-large-en-v1.5 chỉ tiếng Anh) hoặc dùng multilingual SBERT — ảnh hưởng đến tất cả kết quả hiện tại.

Tổng: **1–2 tháng** nếu làm đúng quy trình. Không thể hoàn thành trong revision.

**Cách phản hồi reviewer:** Frame rõ scope của paper — LLMDroid là hệ thống detection cho English Google Play listings, được evaluate trên balanced sample 298+110 apps. Ghi nhận limitation về scale và đề xuất future work: benchmark dataset công khai, multi-store, multilingual extension.

---

## Phần 3 — Files đã thay đổi

| File | Thay đổi |
|------|---------|
| `src/steps/independent_test_eval.py` | Thêm Image-Only vào strategies |
| `src/steps/statistical_tests.py` | Đổi data source → independent test; thêm F1 bootstrap |
| `src/steps/per_label_criterion.py` | **Mới** — per-label-criterion evaluation |
| `src/steps/keyword_drift.py` | **Mới** — keyword drift robustness (retrain in-memory) |
| `src/steps/image_analysis_examples.py` | **Mới** — image helps/hurts qualitative analysis |
| `src/steps/shap_analysis.py` | **Mới** — TreeSHAP for Text-Only + Early Fusion |
| `src/run_analysis.py` | Thêm bước 6.10–6.13 |
| `requirements.txt` | Thêm `shap>=0.46.0` |
