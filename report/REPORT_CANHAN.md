# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Trung Kiên - MSSV: 2A202602764

**Nhóm:** BetBetBet

**Ngày:** 2026-09-19  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) thể hiện hai vector embedding cùng hướng trong không gian ngữ nghĩa đa chiều, biểu thị hai đoạn văn bản có sự tương đồng ngữ nghĩa rất lớn dù có thể dùng câu từ khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên hoàn thành nghĩa vụ đóng học phí trước thời hạn quy định của nhà trường."
- Câu B: "Người học cần nộp tiền học đúng thời gian mà cơ sở đào tạo yêu cầu."
- Tại sao tương đồng: Khác biệt hoàn toàn về từ vựng ("sinh viên" vs "người học", "đóng học phí" vs "nộp tiền học", "nhà trường" vs "cơ sở đào tạo"), nhưng cả hai câu cùng chung một ý nghĩa và mục đích hành động; điều này chứng minh embedding biểu diễn theo ngữ nghĩa chứ không phải so khớp từ khóa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Quy trình phúc khảo điểm thi kết thúc học phần của sinh viên đại học."
- Câu B: "Thực đơn các món ăn trưa hôm nay tại căng tin ký túc xá sinh viên."
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn độc lập (học vụ/khảo thí so với dịch vụ ăn uống), hướng vector của chúng trong không gian embedding phân kỳ (gần như trực giao).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid phụ thuộc vào độ lớn (magnitude/chiều dài) của vector - vốn dễ bị thiên lệch bởi độ dài ngắn của văn bản. Trong khi đó, Cosine similarity chỉ đo góc giữa các vector và chuẩn hóa theo độ dài, giúp phản ánh thuần túy hướng ngữ nghĩa bất kể văn bản dài hay ngắn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> - Bước nhảy giữa các chunk: `step = chunk_size - overlap = 500 - 50 = 450` ký tự.
> - Áp dụng công thức: `số_chunk = ceil((độ_dài - overlap) / (chunk_size - overlap)) = ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.111) = 23`.
> - Kiểm chứng lại bằng repo: `len(FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000))` cho kết quả chính xác 23 chunks.
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, `step = 500 - 100 = 400`, số chunk tăng lên thành `ceil((10000 - 100) / 400) = ceil(9900 / 400) = ceil(24.75) = 25` chunks (tăng thêm 2 chunks). Chúng ta muốn overlap lớn hơn nhằm bảo toàn ngữ cảnh xuyên suốt ranh giới giữa các chunk liền kề, ngăn hiện tượng thông tin hoặc câu văn bị cắt cụt ở biên, giúp tầng retrieval tìm kiếm đầy đủ ngữ cảnh hơn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regex lookbehind `(?<=[.!?])\s+|(?<=\.)\n` để tách ranh giới câu mà vẫn bảo toàn dấu câu kết thúc ở cuối mỗi câu (không bị nuốt dấu). Sau đó gom nhóm tối đa `max_sentences_per_chunk` câu vào mỗi chunk và làm sạch khoảng trắng. Xử lý edge case text rỗng hoặc toàn khoảng trắng trả về `[]`.
> *Edge case đã nhận diện chưa xử lý:* Các từ viết tắt có dấu chấm (như "TS.", "ThS.", "v.v.") hoặc số thập phân (như "3.14") sẽ bị tách nhầm thành dấu kết thúc câu.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán đệ quy 2 chiều theo danh sách separator ưu tiên `["\n\n", "\n", ". ", " ", ""]`: Ưu tiên cắt theo các ranh giới lớn trước (đoạn -> dòng -> câu -> từ -> ký tự); các mảnh nhỏ liền kề được gom lại cho tới sát `chunk_size` để tránh chunk vụn; chỉ khi mảnh nào vượt quá `chunk_size` thì mới đệ quy xuống sâu với separator tiếp theo.
> *Base cases:* (1) `len(current_text) <= chunk_size` trả về `[current_text]`; (2) Hết danh sách separator hoặc gặp separator rỗng `""` thì fallback sang cắt theo ký tự cố định.

**`HeadingChunker.chunk`** — chiến lược riêng của tôi (Custom):
> Tận dụng cấu trúc phân cấp đặc thù của văn bản quy định đại học (các đầu mục `#`, `##`, `###`, và `Điều 1`, `Điều 2`...). Mỗi mục/Điều khoản tạo thành một chunk ngữ nghĩa độc lập, trọn vẹn. Khi một Điều quá dài vượt quá `max_chunk_size=500`, hệ thống dùng `RecursiveChunker` chia nhỏ và **tự động chèn lại tiêu đề Điều vào đầu mỗi mảnh con (Context Re-injection)** để không bị mất ngữ cảnh ở các mảnh tiếp theo.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ in-memory danh sách các bản ghi record được chuẩn hóa qua `_make_record()` (chứa `id`, `content`, `embedding` vector và bản sao độc lập của `metadata` kèm khóa `doc_id`). Khi tìm kiếm, `search()` chuyển tiếp sang helper `_search_records()` để tính cosine similarity bằng tích vô hướng `_dot()` (do embedding đã được chuẩn hóa L2), sắp xếp giảm dần theo `score` và trả về `top_k` kết quả (loại bỏ vector embedding khỏi kết quả để đầu ra hiển thị gọn gàng).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` thực hiện **lọc trước (pre-filtering)** trên tập record theo metadata rồi mới đưa danh sách thỏa mãn vào `_search_records()`. Điều này tránh lỗi nghiêm trọng khi post-filter (lọc sau) có thể trả về 0 kết quả do top-k bị chiếm hết bởi tài liệu không thỏa mãn filter. `delete_document` lọc bỏ tất cả record có `metadata["doc_id"] == doc_id` (xóa toàn bộ các chunk thuộc về file đó) và trả về `True` nếu số lượng phần tử giảm xuống.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Thực hiện theo 3 nhịp RAG: Kiểm tra store rỗng (trả thông báo sớm mà không gọi LLM), truy xuất top-k chunk liên quan nhất từ store, và ghép ngữ cảnh có đánh số thứ tự `[1]`, `[2]`... kèm tên nguồn `source`/`doc_id` để đạt tiêu chí Source Traceability. Prompt áp dụng ràng buộc chống hallucination nghiêm ngặt: chỉ sử dụng dữ kiện trong ngữ cảnh, nói rõ "Không tìm thấy thông tin" nếu thiếu dữ liệu, và yêu cầu trích dẫn số thứ tự tài liệu trước khi chuyển tới `llm_fn`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.10.11, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\kienn\AppData\Local\Programs\Python\Python310\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\kienn\AI20K\K4-DAY07-NguyenTrungKien-2A202602764
plugins: anyio-4.12.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.13s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

*(Đo lường thực tế bằng mô hình OpenAI `text-embedding-3-small` và `compute_similarity`)*

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | Sinh viên nộp học phí qua cổng thanh toán trực tuyến của nhà trường. | Người học chuyển tiền học phí trên hệ thống portal điện tử. | cao | 0.6191 | Đúng |
| 2 | Thời hạn sinh viên nộp đơn phúc khảo điểm thi là 40 ngày. | Sinh viên hoàn thành thủ tục đăng ký học phần trong 7 ngày. | thấp | 0.4883 | Đúng |
| 3 | Ngân hàng OCB hỗ trợ thu hộ học phí trực tiếp cho sinh viên. | Ngân hàng câu hỏi thi kết thúc học phần được bảo mật tuyệt đối. | thấp | 0.4640 | Đúng |
| 4 | Thư viện UEH mở cửa phục vụ bạn đọc từ thứ Hai đến thứ Bảy. | Công tác khám sức khỏe và cấp thuốc định kỳ tại trạm y tế. | thấp | 0.3073 | Đúng |
| 5 | Học bổng khuyến khích học tập chỉ cấp cho sinh viên có điểm rèn luyện tốt trở lên. | Sinh viên bị xếp loại rèn luyện trung bình sẽ không được xét học bổng. | cao | 0.5549 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả đo lường với mô hình embedding thật (`text-embedding-3-small`) phản ánh chính xác ngữ nghĩa sâu:
> 1. Cặp 1 đạt độ tương đồng cao nhất (0.6191) dù sử dụng từ đồng nghĩa ("nộp" vs "chuyển", "cổng thanh toán trực tuyến" vs "hệ thống portal điện tử"), chứng minh embeddings hiểu được sự tương đương về mặt hành động và khái niệm.
> 2. Cặp 4 có điểm thấp nhất (0.3073) vì chủ đề Thư viện và Y tế hoàn toàn tách biệt.
> 3. Cặp 3 cho điểm tương đối thấp (0.4640) mặc dù hai câu cùng xuất hiện từ vựng bề mặt "Ngân hàng" và "sinh viên", chứng tỏ mô hình không bắt chước từ khóa bề mặt (lexical overlap) mà phân biệt rạch ròi giữa tổ chức tài chính tín dụng ("Ngân hàng OCB") và tập hợp đề thi ("Ngân hàng câu hỏi").

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

*Chiến lược của tôi:* `HeadingChunker (mục/Điều khoản, max_chunk=500)` kết hợp mô hình OpenAI `text-embedding-3-small` trên bộ 8 tài liệu quy định UEH (73 chunks, độ dài trung bình 405.7 ký tự).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-----------------|--------------------------------------|:----------:|:------------------------------:|---------------------------------|
| 1 | Mức học bổng khuyến khích học tập loại xuất sắc của UEH bằng bao nhiêu phần trăm suất học bổng toàn phần? | `hoc-bong-khuyen-khich-hoc-tap-76#0` (# Học bổng Khuyến khích học tập...) | 0.7385 | Có | Bằng 150% suất học bổng toàn phần cho SV xuất sắc. |
| 2 | Thời hạn để sinh viên nộp đề nghị phúc khảo điểm thi kết thúc học phần là bao lâu? | `phuc-khao-diem-thi-47#0` (# Phúc khảo điểm thi - Thời hạn...) | 0.7078 | Có | Trong vòng bốn mươi (40) ngày làm việc kể từ ngày thi. |
| 3 | Giờ mở cửa của Thư viện thông minh UEH tại cơ sở Nguyễn Tri Phương (tòa nhà B1 lầu 6) như thế nào? | `93#0` (# Thư viện UEH Giờ mở cửa Thư viện cơ sở NTP...) | 0.7450 | Có | Thứ Hai đến Thứ Sáu: 08:00 - 20:00; Thứ Bảy: 08:00 - 16:00. |
| 4 | Sinh viên có thể đóng học phí qua những hình thức nào theo hướng dẫn của UEH? | `dong-hoc-phi-35#0` (# Đóng học phí Cách 1: Cổng payment...) | 0.6979 | Có | Cổng payment, chuyển khoản ngân hàng, nộp trực tiếp tại OCB. |
| 5 | Sinh viên cần thỏa mãn những điều kiện gì về kết quả học tập và rèn luyện để được xét học bổng khuyến khích? *(Filter: `audience="student"`)* | `hoc-bong-khuyen-khich-hoc-tap-76#1` (# Học bổng Khuyến khích học tập...) | 0.7001 | Có | Kết quả học tập và rèn luyện từ loại Khá trở lên, các môn >= 5. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5 (Đạt 100% Hit@1 và 100% Hit@3)**.
Điểm Cosine Similarity trung bình ở Top-1 đạt **0.718**, cao nhất trong toàn bộ 4 chiến lược thử nghiệm của nhóm.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> 1. Tầm quan trọng cốt tử của việc làm sạch dữ liệu (Data Cleaning): Loại bỏ toàn bộ header, navigation và footer của web trường giúp vector store không bị ô nhiễm bởi các đoạn text rác.
> 2. Sức mạnh của Context Re-injection trong HeadingChunker: Nhờ tự động chèn lại tiêu đề Điều luật vào đầu mỗi chunk con, embedding luôn mang đầy đủ chủ đề gốc, giúp Hit@1 đạt 100% vượt trội so với các phương pháp cắt cứng.
> 3. Sự cần thiết của Metadata Pre-filtering: Khi kho dữ liệu chứa nhiều đối tượng (sinh viên chính quy và cán bộ giảng viên), bắt buộc phải lọc metadata trước khi tìm kiếm để không bị lẫn lộn câu trả lời.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|:----------------:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
