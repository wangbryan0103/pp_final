#include <immintrin.h>
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

#define BEFORE_MAX 0
#define MAX_DIAG 1
#define AFTER_MAX 2

using namespace std;

int gap_score = -7;
int match_score = 5;

void generate_random_seq(string &seqA, int length) {
    const char bases[] = {'A', 'T', 'C', 'G'};
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 3);

    seqA.clear();
    for (int i = 0; i < length; ++i) {
        seqA.push_back(bases[dis(gen)]);
    }
}

void generate_similar_seq(const string &seqA, string &seqB, int length, double similarity) {
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

bool ans_accuracy(int &score_s, int &score_w, int &i_s, int &i_w, int &j_s, int j_w) {
    return (score_s == score_w && i_s == i_w && j_s == j_w);
}

void SmithWaterman_serial(vector<vector<int>>& score, const string seq1, const string seq2, int &max_score_serial, int &max_i_serial, int &max_j_serial) {
    int mismatch_score;
    for (int i = 1; i < score.size(); i++) {
        for (int j = 1; j < score[i].size(); j++) {
            if (seq1[i - 1] == seq2[j - 1]) {
                score[i][j] = score[i - 1][j - 1] + match_score;
            } else {
                if (seq1[i - 1] == 'A' && seq2[j - 1] == 'G' || seq1[i - 1] == 'G' && seq2[j - 1] == 'A' ||
                    seq1[i - 1] == 'C' && seq2[j - 1] == 'T' || seq1[i - 1] == 'T' && seq2[j - 1] == 'C') 
                    mismatch_score = -1;
                else mismatch_score = -3;

                score[i][j] = max({score[i][j - 1] - gap_score, score[i - 1][j] - gap_score, 
                                   score[i - 1][j - 1] + mismatch_score, 0});
            }
            if (score[i][j] > max_score_serial) {
                max_i_serial = i;
                max_j_serial = j;
                max_score_serial = score[i][j];
            }
        }
    }
}

// SIMD-optimized function for calculating scores in parallel
void calculate_simd(const string &seqA, const string &seqB, const vector<int> &prev1, 
                    const vector<int> &prev2, vector<int> &curr, int len_diag, int diag_offset) {
    const __m256i gap_vec = _mm256_set1_epi32(gap_score);
    const __m256i match_vec = _mm256_set1_epi32(match_score);
    const __m256i mismatch1_vec = _mm256_set1_epi32(-1);
    const __m256i mismatch3_vec = _mm256_set1_epi32(-3);

    for (int d = 0; d < len_diag; d += 8) { // Process 8 elements at a time
        int i = diag_offset + d;
        int j = d;
        
        __m256i curr_vec = _mm256_setzero_si256();
        __m256i diag_vec = _mm256_loadu_si256((__m256i*)&prev1[d]); // Load diagonal elements
        __m256i left_vec = _mm256_loadu_si256((__m256i*)&prev2[d]); // Load left elements
        __m256i up_vec = _mm256_loadu_si256((__m256i*)&prev2[d + 1]); // Load up elements

        // Calculate match/mismatch
        __m256i match_mask = _mm256_set1_epi32(seqA[i - 1] == seqB[j - 1] ? 1 : 0);
        __m256i score_match = _mm256_blendv_epi8(mismatch1_vec, match_vec, match_mask);

        curr_vec = _mm256_max_epi32(curr_vec, _mm256_add_epi32(diag_vec, score_match));
        curr_vec = _mm256_max_epi32(curr_vec, _mm256_add_epi32(left_vec, gap_vec));
        curr_vec = _mm256_max_epi32(curr_vec, _mm256_add_epi32(up_vec, gap_vec));

        _mm256_storeu_si256((__m256i*)&curr[d], curr_vec); // Store results
    }
}

void SmithWaterman_parallel(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    vector<int> prev1(rows, 0);
    vector<int> prev2(rows, 0);
    vector<int> curr(rows, 0);

    #pragma omp parallel
    {
        int local_max_score = 0, local_i = 0, local_j = 0;

        for (int diag = 2; diag < rows + cols - 1; ++diag) {
            int start_row = max(0, diag - (rows - 1)); 
            int end_row = min(diag, rows - 1);
            int len_diag = end_row - start_row + 1;

            #pragma omp for
            for (int d = 0; d < len_diag; d += 8) {
                calculate_simd(seqA, seqB, prev1, prev2, curr, len_diag, d);
            }

            #pragma omp barrier
            #pragma omp single
            {
                prev1.swap(prev2);
                prev2.swap(curr);
                fill(curr.begin(), curr.end(), 0);
            }
        }

        #pragma omp critical
        {
            if (local_max_score > max_score) {
                max_score = local_max_score;
                max_i = local_i;
                max_j = local_j;
            }
        }
    }
}

int main(int argc, char **argv) {
    string seqA, seqB;
    int length = 20000;
    double similarity = 0.7;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);

    int max_score_parallel = 0, max_i_parallel = 0, max_j_parallel = 0;
    auto start_parallel = chrono::high_resolution_clock::now();
    SmithWaterman_parallel(seqA, seqB, max_score_parallel, max_i_parallel, max_j_parallel);
    auto end_parallel = chrono::high_resolution_clock::now();
    auto duration_parallel = chrono::duration_cast<chrono::nanoseconds>(end_parallel - start_parallel);

    cout << "Parallel Score : " << max_score_parallel << endl;
    cout << "Parallel Time: " << duration_parallel.count() << " ns" << endl;
    return 0;
}
