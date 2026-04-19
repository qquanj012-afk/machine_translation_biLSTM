# process_mrpc.py
# Script chuyển đổi file MSR Paraphrase Corpus sang định dạng raw cho mô hình

import os


def process_mrpc_file(input_path, output_path):
    """
    Đọc file MRPC, lọc các cặp paraphrase (Quality=1) và ghi ra file output
    Mỗi dòng: câu_gốc \t câu_paraphrase
    """
    with open(input_path, 'r', encoding='utf8') as f_in:
        # Bỏ qua dòng tiêu đề
        header = f_in.readline()
        print(f"Header: {header.strip()}")

        line_count = 0
        kept_count = 0

        with open(output_path, 'w', encoding='utf8') as f_out:
            for line in f_in:
                line_count += 1
                parts = line.strip().split('\t')
                # Cấu trúc: Quality #1 ID #2 ID #1 String #2 String
                if len(parts) >= 5:
                    quality = parts[0]
                    sent1 = parts[3]
                    sent2 = parts[4]
                    if quality == '1':  # chỉ lấy paraphrase
                        f_out.write(f"{sent1}\t{sent2}\n")
                        kept_count += 1
                else:
                    print(f"Dòng {line_count} không đúng định dạng: {line[:100]}")

    print(f"Đã xử lý {line_count} dòng, giữ lại {kept_count} cặp paraphrase.")
    print(f"Kết quả lưu tại: {output_path}")


if __name__ == "__main__":
    # Đường dẫn trong project
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, "data", "raw")

    train_input = os.path.join(raw_dir, "msr_paraphrase_train.txt")
    test_input = os.path.join(raw_dir, "msr_paraphrase_test.txt")

    train_output = os.path.join(raw_dir, "paraphrase_train.txt")
    test_output = os.path.join(raw_dir, "paraphrase_test.txt")

    print("=== Xử lý tập train ===")
    process_mrpc_file(train_input, train_output)

    print("\n=== Xử lý tập test ===")
    process_mrpc_file(test_input, test_output)

    print("\nHoàn thành! File raw đã sẵn sàng.")