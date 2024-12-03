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


void SmithWaterman_SIMD(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j, vector<vector<int>> &simd_score) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    // 初始化用於儲存每個位置分數的矩陣
    simd_score.assign(rows, vector<int>(cols, 0));

    // 使用對齊分配記憶體
    int *prev = (int *)_mm_malloc(cols * sizeof(int), 32);
    int *curr = (int *)_mm_malloc(cols * sizeof(int), 32);

    // 初始化
    fill(prev, prev + cols, 0);
    fill(curr, curr + cols, 0);

    for (int diag = 2; diag < rows + cols - 1; ++diag) {
        int start_row = max(1, diag - (cols - 1));
        int end_row = min(rows - 1, diag - 1);
        int len_diag = end_row - start_row + 1;

        for (int d = 0; d < len_diag; d += 8) {
            int remaining = min(8, len_diag - d);

            // 設定 SIMD 處理的當前區域
            __m256i seqA_chars = _mm256_set_epi32(
                seqA[start_row + d + 7 - 1], seqA[start_row + d + 6 - 1],
                seqA[start_row + d + 5 - 1], seqA[start_row + d + 4 - 1],
                seqA[start_row + d + 3 - 1], seqA[start_row + d + 2 - 1],
                seqA[start_row + d + 1 - 1], seqA[start_row + d - 1]);

            __m256i seqB_chars = _mm256_set_epi32(
                seqB[diag - start_row - d - 7 - 1], seqB[diag - start_row - d - 6 - 1],
                seqB[diag - start_row - d - 5 - 1], seqB[diag - start_row - d - 4 - 1],
                seqB[diag - start_row - d - 3 - 1], seqB[diag - start_row - d - 2 - 1],
                seqB[diag - start_row - d - 1 - 1], seqB[diag - start_row - d - 1]);

            // 計算匹配分數
            __m256i match_mask = _mm256_cmpeq_epi32(seqA_chars, seqB_chars);
            __m256i mismatch_score_vec = _mm256_blendv_epi8(_mm256_set1_epi32(GAP_SCORE), _mm256_set1_epi32(MATCH_SCORE), match_mask);

            // 加載之前的分數
            __m256i diag_scores = _mm256_loadu_si256((__m256i *)&prev[d]);
            __m256i up_scores = _mm256_loadu_si256((__m256i *)&prev[d + 1]);
            __m256i left_scores = _mm256_loadu_si256((__m256i *)&curr[d]);

            // 計算當前分數
            __m256i current_scores = _mm256_add_epi32(diag_scores, mismatch_score_vec);
            current_scores = _mm256_max_epi32(current_scores, _mm256_max_epi32(up_scores, left_scores));
            current_scores = _mm256_max_epi32(current_scores, _mm256_setzero_si256());

            // 儲存當前分數
            _mm256_storeu_si256((__m256i *)&curr[d], current_scores);

            // 更新矩陣並更新最大分數
            for (int k = 0; k < remaining; ++k) {
                int scalar_score = curr[d + k];
                simd_score[start_row + d + k][diag - start_row - d - k] = scalar_score;

                if (scalar_score > max_score) {
                    max_score = scalar_score;
                    max_i = start_row + d + k;
                    max_j = diag - start_row - d - k;
                }
            }
        }

        // 更新行緩衝區
        swap(prev, curr);
        fill(curr, curr + cols, 0);
    }

    // 正確釋放內存
    _mm_free(prev);
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
    cout << "    ";
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
    cout << "    ";
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
    cout << "    ";
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