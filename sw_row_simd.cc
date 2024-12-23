#include <ctime>
#include <omp.h>
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
#define MISMATCH_SCORE -3
#define GAP_SCORE -7
#define BEFORE_MAX 0
#define NEXT_MAX 1
#define AFTER_MAX 2
#define ALIGNMENT 32

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



void SmithWaterman_serial(vector<vector<int>> &score, const string &seqA, const string &seqB,
                          int &max_score_serial, int &max_i_serial, int &max_j_serial) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            int diag_score = (seqA[i - 1] == seqB[j - 1]) ? MATCH_SCORE : MISMATCH_SCORE;
                diag_score = score[i - 1][j - 1] + diag_score;

            int up_score = score[i - 1][j] + GAP_SCORE;
            int left_score = score[i][j - 1] + GAP_SCORE;


            score[i][j] = max({diag_score, up_score, left_score, 0});


            if (score[i][j] > max_score_serial) {
                max_score_serial = score[i][j];
                max_i_serial = i;
                max_j_serial = j;
            }
        }
    }
}


void SmithWaterman_diag(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;


    vector<int> prev1(cols, 0);
    vector<int> prev2(cols, 0);
    vector<int> curr(cols, 0);


    for (int diag = 2; diag < rows + cols - 1; ++diag) {
        int start_row = max(1, diag - (cols - 1));
        int end_row = min(rows - 1, diag - 1);
        int len_diag = end_row - start_row + 1;

        for (int d = 0; d < len_diag; ++d) {
            int i = start_row + d;
            int j = diag - i;

            if (i == 0 || j == 0) {
                curr[j] = 0;
            } else {
                int diag_score = (seqA[i - 1] == seqB[j - 1]) ? MATCH_SCORE : MISMATCH_SCORE;
                    diag_score = prev1[j - 1] + diag_score;
                int up_score = prev2[j] + GAP_SCORE;
                int left_score = prev2[j - 1] + GAP_SCORE;

                curr[j] = max({diag_score, up_score, left_score, 0});
            }


            if (curr[j] > max_score) {
                max_score = curr[j];
                max_i = i;
                max_j = j;
            }
        }

        // 更新波前數據
        prev1.swap(prev2);
        prev2.swap(curr);
        fill(curr.begin(), curr.end(), 0);
    }
}



void SmithWaterman_SIMD(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j, vector<vector<int>> &score) {

    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    char * seqA_buffer = static_cast<char *>(aligned_alloc(ALIGNMENT, rows));
    char * seqB_buffer = static_cast<char *>(aligned_alloc(ALIGNMENT, cols));

    std::memcpy(seqA_buffer, seqA.c_str(), rows);
    std::memcpy(seqB_buffer, seqB.c_str(), cols);
    __m256i test    = _mm256_set1_epi32(1);
    //__m256i test1    = _mm256_set1_epi32(-1);
    __m256i test2    = _mm256_set1_epi32(0);
    __m256i match_score    = _mm256_set1_epi32(MATCH_SCORE);
    __m256i mismatch_score = _mm256_set1_epi32(MISMATCH_SCORE);
    __m256i gap_score      = _mm256_set1_epi32(GAP_SCORE);
    for (int i = 1; i < rows; ++i) {
        __m256i last = test2;
        for (int j = 1; j < cols; j += 8) {

            __m256i seqA_chars = _mm256_set1_epi32((int)seqA_buffer[i-1]);
            __m256i seqB_chars = _mm256_set_epi32(
            (int)seqB_buffer[j + 6],
            (int)seqB_buffer[j + 5],
            (int)seqB_buffer[j + 4],
            (int)seqB_buffer[j + 3],
            (int)seqB_buffer[j + 2],
            (int)seqB_buffer[j + 1],
            (int)seqB_buffer[j],
            (int)seqB_buffer[j -1]);

            __m256i match_mask = _mm256_cmpeq_epi8(seqA_chars, seqB_chars);
                    match_mask = _mm256_and_si256(test, match_mask);
                    match_mask = _mm256_xor_si256(match_mask, _mm256_set1_epi8(0xFF));
                    match_mask = _mm256_add_epi32(match_mask, test); 

            __m256i diag_score = _mm256_blendv_epi8(mismatch_score, match_score, match_mask);
           
            diag_score = _mm256_add_epi32( _mm256_loadu_si256((__m256i *)&score[i-1][j-1]), diag_score);

            __m256i up_score   = _mm256_add_epi32( _mm256_loadu_si256((__m256i *)&score[i-1][ j ]), gap_score);
            
            __m256i temp_score = _mm256_max_epi32(up_score, diag_score);
            
            temp_score = _mm256_max_epi32(test2 , temp_score);
            
            __m256i left_vec = _mm256_slli_si256(temp_score, 4);
            __m256i permuted = _mm256_permute2x128_si256(temp_score, temp_score, 10);
            __m256i tmp = _mm256_srli_si256(permuted, 12);
            __m256i test = _mm256_add_epi32(left_vec,tmp);
            
            __m256i left_minus_penalty = _mm256_add_epi32(test, gap_score);

            __m256i result_vec = _mm256_max_epi32(temp_score, left_minus_penalty);

            __m256i lefttmp =_mm256_slli_si256(result_vec,4);
            __m256i permuted_tmp = _mm256_permute2x128_si256(result_vec, result_vec, 10);
            __m256i tmp_tmp = _mm256_srli_si256(permuted_tmp, 12);
            __m256i test_tmp = _mm256_add_epi32(lefttmp,tmp_tmp);
            
            test_tmp = _mm256_add_epi32(test_tmp,last);
            /*alignas(32) int32_t buffer[8]; 
            _mm256_storeu_si256((__m256i*)buffer, test_tmp);

            std::cout << "Elements: ";
            for (int i = 0; i < 8; ++i) {
                std::cout << buffer[i] << " ";
            }
            std::cout << std::endl;*/
            __m256i left_minus_penalty_tmp = _mm256_add_epi32(test_tmp, gap_score);
            
            
            __m256i result_vec_final = _mm256_max_epi32(result_vec, left_minus_penalty_tmp);
            _mm256_storeu_si256((__m256i*)&score[i][j], result_vec_final);
            __m256i idx = _mm256_set_epi32(-1, 0, 0, 0, 0, 0, 0, 0);
            last = test2;
            last = _mm256_and_si256(result_vec_final, idx);

            last = _mm256_permute2x128_si256(last, last, 1);
            last = _mm256_srli_si256(last, 12);
        }
        /*for (int k = 1; k < cols; k += 8) {
                __m256i resulttmp = _mm256_loadu_si256((__m256i*)&score[i][k]);
                __m256i left_vec = _mm256_loadu_si256((__m256i*)&score[i][k-1]);

                __m256i left_minus_penalty = _mm256_add_epi32(left_vec, gap_score);
                __m256i result_vec = _mm256_max_epi32(resulttmp, left_minus_penalty);
                _mm256_storeu_si256((__m256i*)&score[i][k], result_vec);
        }*/
        for(int j=1;j<cols;j++){
             if (score[i][j] > max_score) {
                max_score = score[i][j];
                max_i = i;
                max_j = j;
            
            }
        }
    }
}




int main(int argc, char **argv) {

    // ====================
    // generate sequence
    // ====================
    //string seqA  = "GATCTCGTGAGATCAC";
    //string seqB  = "GATAGCATCCAGTCAA";
    string seqA,seqB;

    int length = 10000;
    double similarity = 0.7;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);

    //  =====================================================
    //  parallel version of wavefront with two anti-diagonals
    //  =====================================================
    int max_score_diag = 0, max_i_diag = 0, max_j_diag = 0;

    auto start_diag = chrono::high_resolution_clock::now();
    SmithWaterman_diag(seqA, seqB, max_score_diag, max_i_diag, max_j_diag);
    auto end_diag = chrono::high_resolution_clock::now();
    auto duration_diag = chrono::duration_cast<chrono::nanoseconds>(end_diag - start_diag);


    //  =====================================================
    //  serial version for accuracy
    //  =====================================================
    int max_score_serial = 0, max_i_serial = 0, max_j_serial = 0;


    vector<vector<int>> score_serial(seqA.size() + 1, vector<int>(seqB.size() + 1, 0));
    auto start_serial = chrono::high_resolution_clock::now();
    //SmithWaterman_serial(seqA, seqB, max_score_serial, max_i_serial, max_j_serial);
    SmithWaterman_serial(score_serial,seqA,seqB, max_score_serial, max_i_serial, max_j_serial);
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::nanoseconds>(end_serial - start_serial);
    //  =====================================================
    //  SIMD version
    //  =====================================================
    vector<vector<int>> score_SIMD(seqA.size() + 1, vector<int>(seqB.size() + 1, 0));
    int max_score_simd = 0, max_i_simd = 0, max_j_simd = 0;
    //reverse(seqB.begin(), seqB.end());
    auto start_simd = chrono::high_resolution_clock::now();
    SmithWaterman_SIMD(seqA, seqB, max_score_simd, max_i_simd, max_j_simd, score_SIMD);
    auto end_simd = chrono::high_resolution_clock::now();
    auto duration_simd = chrono::duration_cast<chrono::nanoseconds>(end_simd - start_simd);
    /*cout<<"  ";
    for(int k=0;k<seqB.length();k++)
        cout<<seqB[k]<<"  ";
	    cout<<endl;
        for(int i=1;i<score_SIMD.size();i++){
		    cout<<seqA[i-1]<<"  ";
                for (int j=1;j<score_SIMD[i].size();j++){
			        cout<<score_SIMD[i][j]<<"  ";
		}
		cout<<endl;
	}*/

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

