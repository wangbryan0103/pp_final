#include <ctime>
#include <omp.h>
#include <cstdlib>
#include <random>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> 
#include <chrono>
#include <utility>
#include <immintrin.h>
#include <iomanip>
#include <cstring>

#define MATCH_SCORE 5
#define GAP_SCORE -7
#define BEFORE_MAX 0
#define NEXT_MAX 1
#define AFTER_MAX 2

using namespace std;


void generate_random_seq(string &seqA, int length){
    const char bases[] = {'A', 'T', 'C', 'G'};
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 3);

    seqA.clear();
    for (int i = 0; i < length; ++i) {
        seqA.push_back(bases[dis(gen)]);
    }
}

void generate_similar_seq(const string &seqA, string &seqB, int length, double similarity){
    const char bases[] = {'A', 'T', 'C', 'G'};
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> normal_dis(0, 3);
    uniform_real_distribution<> change(0.0, 1.0);

    seqB.clear();
    for (int i = 0; i < length; ++i) {
        if (change(gen) > similarity) {
            seqB.push_back(bases[normal_dis(gen)]);
        } else {
            seqB.push_back(seqA[i]);
        }
    }
}

bool ans_accuracy(int &score_s, int &score_w, int &i_s, int &i_w, int &j_s, int j_w){
    return (score_s == score_w && i_s == i_w && j_s == j_w);
}



void SmithWaterman_serial(std::vector<std::vector<int>> &score, const std::string &seqA, const std::string &seqB,
                          int &max_score_serial, int &max_i_serial, int &max_j_serial) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            int match_score = (seqA[i - 1] == seqB[j - 1]) ? MATCH_SCORE : GAP_SCORE;

            // 計算來自左上角、上方和左方的得分
            int diag_score = score[i - 1][j - 1] + match_score;
            int up_score = score[i - 1][j] + GAP_SCORE;
            int left_score = score[i][j - 1] + GAP_SCORE;

            // 更新當前格子的分數
            score[i][j] = std::max({diag_score, up_score, left_score, 0});

            // 更新最大分數及其位置
            if (score[i][j] > max_score_serial) {
                max_score_serial = score[i][j];
                max_i_serial = i;
                max_j_serial = j;
            }
        }
    }
}


void SmithWaterman_diag(const std::string &seqA, const std::string &seqB, int &max_score, int &max_i, int &max_j, std::vector<std::vector<int>> &diag_score) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    // 初始化用於儲存每個位置分數的矩陣
    diag_score.assign(rows, std::vector<int>(cols, 0));

    // 初始化計算所需的向量
    std::vector<int> prev1(cols, 0);
    std::vector<int> prev2(cols, 0);
    std::vector<int> curr(cols, 0);

    for (int diag = 2; diag < rows + cols - 1; ++diag) {
        int start_row = std::max(1, diag - (cols - 1));
        int end_row = std::min(rows - 1, diag - 1);
        int len_diag = end_row - start_row + 1;

        for (int d = 0; d < len_diag; ++d) {
            int i = start_row + d;
            int j = diag - i;

            if (i == 0 || j == 0) {
                curr[j] = 0;
            } else {
                int mismatch_score = (seqA[i - 1] == seqB[j - 1]) ? MATCH_SCORE : GAP_SCORE;
                int diag_score = prev1[j - 1] + mismatch_score;
                int up_score = prev2[j] + GAP_SCORE;
                int left_score = prev2[j - 1] + GAP_SCORE;

                curr[j] = std::max({diag_score, up_score, left_score, 0});
            }

            // 儲存分數到 diag_score
            diag_score[i][j] = curr[j];

            if (curr[j] > max_score) {
                max_score = curr[j];
                max_i = i;
                max_j = j;
            }
        }

        // 更新波前數據
        prev1.swap(prev2);
        prev2.swap(curr);
        std::fill(curr.begin(), curr.end(), 0);
    }
}

// 2. string reverse seqA, rev to copy 1 times (load) , speedup
void SmithWaterman_SIMD(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j, vector<vector<int>> &simd_score) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    // 初始化用於儲存每個位置分數的矩陣
    simd_score.assign(rows, vector<int>(cols, 0));

    // 使用對齊分配記憶體，額外分配空間處理動態偏移
    int *prev2 = (int *)_mm_malloc((cols + 1) * sizeof(int), 32); 
    int *prev1 = (int *)_mm_malloc((cols + 1) * sizeof(int), 32); 
    int *curr = (int *)_mm_malloc((cols + 1) * sizeof(int), 32); 

    // 初始化緩衝區
    fill(prev2, prev2 + cols + 1, 0);
    fill(prev1, prev1 + cols + 1, 0);
    fill(curr, curr + cols + 1, 0);

    char seqA_buffer[rows + 7] = {0}; // 增加 7 個填充位元，避免越界
    char seqB_buffer[cols + 7] = {0};
    strncpy(seqA_buffer + 7, seqA.c_str(), seqA.size());
    // 將 seqB 倒序填充到緩衝區
    for (size_t i = 0; i < seqB.size(); ++i) {
        seqB_buffer[i] = seqB[seqB.size() - 1 - i];
    }
    for (int i = 0; i < 7; i++){
        seqB_buffer[cols+i-1] = '_';
    }

    for (int diag = 2; diag < rows + cols - 1; ++diag) {
        int start_row = max(1, diag - (cols - 1));
        int end_row = min(rows - 1, diag - 1);
        int len_diag = end_row - start_row + 1;

        for (int d = 0; d < len_diag; d += 8) {
            int remaining = min(8, len_diag - d);

            // 使用 _mm256_loadu_si256 一次抓取
            int indexA = start_row - 1 + 7 + d;
            int indexB = cols - 1 - end_row + d;
            const char *seqA_ptr = &seqA_buffer[indexA];
            const char *seqB_ptr = &seqB_buffer[indexB];
            __m256i seqA_chars = _mm256_loadu_si256((__m256i *)seqA_ptr);
            __m256i seqB_chars = _mm256_loadu_si256((__m256i *)seqB_ptr);

            // 打印抓取的內容
            cout << "Diag: " << diag << ", Offset: " << d << endl;

            char seqA_extracted[8];
            char seqB_extracted[8];
            _mm256_storeu_si256((__m256i *)seqA_extracted, seqA_chars);
            _mm256_storeu_si256((__m256i *)seqB_extracted, seqB_chars);

            cout << "Loaded seqA: ";
            for (int k = 0; k < remaining; ++k) {
                cout << seqA_extracted[k] << " ";
            }
            cout << endl;

            cout << "Loaded seqB: ";
            for (int k = 0; k < remaining; ++k) {
                cout << seqB_extracted[k] << " ";
            }
            cout << endl;

            // 計算 match_mask
            __m256i match_mask = _mm256_cmpeq_epi8(seqA_chars, seqB_chars);

            // 打印 match_mask 的內容
            unsigned char mask_extracted[8];
            _mm256_storeu_si256((__m256i *)mask_extracted, match_mask);

            cout << "Match Mask: ";
            for (int k = 0; k < 8; ++k) {
                cout << (mask_extracted[k] ? "1" : "0") << " "; // 1 表示匹配，0 表示不匹配
            }
            cout << endl;

            cout << "Match Comparisons: "<< endl;
            for (int k = 0; k < 8; ++k) {
                cout << "seqA[" << k << "] = " << seqA_extracted[k]
                    << ", seqB[" << k << "] = " << seqB_extracted[k]
                    << ", Match: " << ((mask_extracted[k] != 0) ? "Yes" : "No") << endl;
            }

            // 將 8 位元的 match_mask 擴展為 32 位元
            __m256i extended_mask = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(match_mask));

            // 計算 match_scores，確保與 diag_scores 格式一致
            __m256i match_scores = _mm256_or_si256(
                _mm256_and_si256(extended_mask , _mm256_set1_epi32(MATCH_SCORE)),  // 匹配的為 MATCH_SCORE
                _mm256_andnot_si256(extended_mask , _mm256_set1_epi32(GAP_SCORE)) // 不匹配的為 GAP_SCORE
            );

            // 動態偏移量計算
            int disp = min(1, max(0, diag - cols));
            int diagDisp = min(2, max(0, diag - cols));
            
            // 動態計算分數
            __m256i diag_scores = _mm256_add_epi32(
                _mm256_loadu_si256((__m256i *)&prev2[d + diagDisp]), match_scores);
            __m256i up_scores = _mm256_add_epi32(
                _mm256_loadu_si256((__m256i *)&prev1[d + 1 + disp]), _mm256_set1_epi32(GAP_SCORE));
            __m256i left_scores = _mm256_add_epi32(
                _mm256_loadu_si256((__m256i *)&prev1[d + disp]), _mm256_set1_epi32(GAP_SCORE));

            int diag_values[cols + 1], up_values[cols + 1], left_values[cols + 1];
            int match_values[cols + 1];
            _mm256_storeu_si256((__m256i *)match_values, match_scores);
            _mm256_storeu_si256((__m256i *)diag_values, diag_scores);
            _mm256_storeu_si256((__m256i *)up_values, up_scores);
            _mm256_storeu_si256((__m256i *)left_values, left_scores);

            cout << "Match Scores: ";
            for (int k = 0; k < 8; ++k) cout << match_values[k] << " ";
            cout << endl;

            cout << "Diag Scores: ";
            for (int k = 0; k < 8; ++k) cout << diag_values[k] << " ";
            cout << endl;

            cout << "Up Scores: ";
            for (int k = 0; k < 8; ++k) cout << up_values[k] << " ";
            cout << endl;

            cout << "Left Scores: ";
            for (int k = 0; k < 8; ++k) cout << left_values[k] << " ";
            cout << endl;

            // 計算當前分數
            __m256i current_scores = _mm256_max_epi32(diag_scores, _mm256_max_epi32(up_scores, left_scores));
            current_scores = _mm256_max_epi32(current_scores, _mm256_setzero_si256());

            // 儲存當前分數到緩衝區
            _mm256_storeu_si256((__m256i *)&curr[1 + d], current_scores);

            // 更新矩陣並檢查最大分數
            for (int k = 0; k < remaining; ++k) {
                int scalar_score = curr[d + k+1];
                simd_score[start_row + d + k][diag - start_row - d - k] = scalar_score;

                if (scalar_score > max_score) {
                    max_score = scalar_score;
                    max_i = start_row + d + k;
                    max_j = diag - start_row - d - k;
                }
            }
        }

        cout << "Prev2: ";
        for (int k = 0; k < cols + 1; ++k) {
            cout << prev2[k] << " ";
        }
        cout << endl;

        cout << "Prev1: ";
        for (int k = 0; k < cols + 1; ++k) {
            cout << prev1[k] << " ";
        }
        cout << endl;

        cout << "Curr Scores: ";
        for (int k = 0; k < cols + 1; ++k) cout << curr[k] << " ";
        cout << endl;

        // 更新波前數據
        int *temp = prev2;
        prev2 = prev1;
        prev1 = curr;
        curr = temp;

        fill(curr, curr + cols +1, 0); // 重置當前緩衝區
    }

    // 正確釋放內存
    _mm_free(prev2);
    _mm_free(prev1);
    _mm_free(curr);
}




int main(int argc, char **argv) {

    // ====================
    // generate sequence
    // ====================
    string seqA; // = "GATCTCGT";
    string seqB; // = "GATAGCAT";


    int length = 10;
    double similarity = 0.7;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);
    // seqA = "ATTGTGTCGC";
    // seqB = "TTTGTGTCGC";

    cout << "Sequence A: " << seqA << endl;
    cout << "Sequence B: " << seqB << endl;

    //  =====================================================
    //  parallel version of wavefront with two anti-diagonals
    //  =====================================================
    int max_score_diag = 0, max_i_diag = 0, max_j_diag = 0;
    std::vector<std::vector<int>> diag_score;

    auto start_diag = chrono::high_resolution_clock::now();
    SmithWaterman_diag(seqA, seqB, max_score_diag, max_i_diag, max_j_diag, diag_score);
    auto end_diag = chrono::high_resolution_clock::now();
    auto duration_diag = chrono::duration_cast<chrono::nanoseconds>(end_diag - start_diag);


    //  =====================================================
    //  serial version for accuracy
    //  =====================================================
    int max_score_serial = 0, max_i_serial = 0, max_j_serial = 0;


    std::vector<std::vector<int>> serial_score(seqA.size() + 1, std::vector<int>(seqB.size() + 1, 0));
    auto start_serial = chrono::high_resolution_clock::now();
    //SmithWaterman_serial(seqA, seqB, max_score_serial, max_i_serial, max_j_serial);
	SmithWaterman_serial(serial_score,seqA,seqB, max_score_serial, max_i_serial, max_j_serial);
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::nanoseconds>(end_serial - start_serial);

    //  =====================================================
    //  SIMD version
    //  =====================================================
    int max_score_simd = 0, max_i_simd = 0, max_j_simd = 0;
    vector<vector<int>> simd_score;

    auto start_simd = chrono::high_resolution_clock::now();
    SmithWaterman_SIMD(seqA, seqB, max_score_simd, max_i_simd, max_j_simd, simd_score);
    auto end_simd = chrono::high_resolution_clock::now();
    auto duration_simd = chrono::duration_cast<chrono::nanoseconds>(end_simd - start_simd);


    // Print serial table
    cout << "\nSerial Score Table:" << endl;
    cout << "      ";
    for (char c : seqB) cout << setw(4) << c;
    cout << endl;
    for (size_t i = 0; i <= seqA.size(); ++i) {
        if (i > 0) cout << seqA[i - 1] << " ";
        else cout << "  ";
        for (size_t j = 0; j <= seqB.size(); ++j) {
            cout << setw(4) << serial_score[i][j];
        }
        cout << endl;
    }

    // Print diagonal table
    cout << "\nDiagonal Calculation Table:" << endl;
    cout << "      ";
    for (char c : seqB) cout << setw(4) << c;
    cout << endl;
    for (size_t i = 0; i <= seqA.size(); ++i) {
        if (i > 0) cout << seqA[i - 1] << " ";
        else cout << "  ";
        for (size_t j = 0; j <= seqB.size(); ++j) {
            cout << setw(4) << diag_score[i][j];
        }
        cout << endl;
    }

    // Print SIMD table
    cout << "\nSIMD Score Table:" << endl;
    cout << "      ";
    for (char c : seqB) cout << setw(4) << c;
    cout << endl;
    for (size_t i = 0; i <= seqA.size(); ++i) {
        if (i > 0) cout << seqA[i - 1] << " ";
        else cout << "  ";
        for (size_t j = 0; j <= seqB.size(); ++j) {
            cout << setw(4) << simd_score[i][j];
        }
        cout << endl;
    }

    
    // Accuracy test of diag
    if (ans_accuracy(max_score_serial, max_score_diag, max_i_serial, max_i_diag, max_j_serial, max_j_diag))
        cout << "| diag score_test passed |" << endl;
    else
        cout << "| diag score_test failed |" << endl;
    
    // Accuracy test of SIMD
    if (ans_accuracy(max_score_serial, max_score_simd, max_i_serial, max_i_simd, max_j_serial, max_j_simd))
        cout << "| SIMD score_test passed |" << endl << endl;
    else
        cout << "| SIMD score_test failed |" << endl << endl;

    // Output max scores and execution times
    cout << "Serial Score   : " << max_score_serial << " at (" << max_i_serial << ", " << max_j_serial << ")" << endl;
    cout << "Diagonal Score : " << max_score_diag << " at (" << max_i_diag << ", " << max_j_diag << ")" << endl;
    cout << "SIMD Score : " << max_score_simd << " at (" << max_i_simd << ", " << max_j_simd << ")" << endl << endl;

    cout << "==========  serial  ==========" << endl;
    cout << "time: " << duration_serial.count() << " ns" << endl;
    cout << "========== Diagonal ==========" << endl;
    cout << "time: " << duration_diag.count() << " ns" << endl;
    cout << "==========   SIMD   ==========" << endl;
    cout << "time: " << duration_simd.count() << " ns" << endl << endl;

    cout << "Speedup of Diag v.s. Serial: " << (double)duration_serial.count() / (double)duration_diag.count() << endl;
    cout << "Speedup of SIMD v.s. Serial: " << (double)duration_serial.count() / (double)duration_simd.count() << endl;
    cout << "Speedup of SIMD v.s. Diag: " << (double)duration_diag.count() / (double)duration_simd.count() << endl;

    return 0;
}
/*
make METHOD=m7; srun -c 1 ./sw
*/