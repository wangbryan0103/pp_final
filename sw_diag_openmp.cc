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

#define MATCH_SCORE 5
#define MISMATCH_SCORE -3
#define GAP_SCORE -7
#define BEFORE_MAX 0
#define NEXT_MAX 1
#define AFTER_MAX 2
#define ALIGNMENT 32

using namespace std;


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

bool ans_accuracy(int &score_s, int &score_w, int &i_s, int &i_w, int &j_s, int j_w)
{
    return (score_s == score_w && i_s == i_w && j_s == j_w);
}



void SmithWaterman_serial_table(vector<vector<int>> &score, const string &seqA, const string &seqB,
                                    int &maxScore_serial_row, int &maxi_serial_row, int &maxj_serial_row) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            int diag_score = (seqA[i - 1] == seqB[j - 1]) ? MATCH_SCORE : MISMATCH_SCORE;
                diag_score = score[i - 1][j - 1] + diag_score;

            int up_score   = score[i - 1][j] + GAP_SCORE;
            int left_score = score[i][j - 1] + GAP_SCORE;
            
            score[i][j] = max({diag_score, up_score, left_score, 0});
            
            if (score[i][j] > maxScore_serial_row) {
                maxScore_serial_row = score[i][j];
                maxi_serial_row = i;
                maxj_serial_row = j;
            }
        }
    }
}


void SmithWaterman_serial_diag(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;


    
    vector<int> prev1(cols, 0);
    vector<int> prev2(cols, 0);
    
    for (int diag = 2; diag < rows + cols - 1; ++diag) {
        vector<int> curr(cols, 0);    
        int start_row = max(1, diag - (cols - 1));
        int end_row   = min(rows - 1, diag - 1);
        int len_diag  = end_row - start_row + 1;

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

        prev1.swap(prev2);
        prev2.swap(curr);
        fill(curr.begin(), curr.end(), 0);
    }
}




//assune len_A > len_B
void SmithWaterman_parallel_diag(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    vector<int> prev1(cols, 0);
    vector<int> prev2(cols, 0);
    
    for (int diag = 2; diag < rows + cols - 1; ++diag) {
        vector<int> curr(cols, 0);    
        int start_row = max(1, diag - (cols - 1));
        int end_row   = min(rows - 1, diag - 1);
        int len_diag  = end_row - start_row + 1;

        #pragma omp parallel
        {
            int local_max = 0;
            int local_i, local_j;

            #pragma omp for 
            for (int d = 0; d < len_diag; ++d) {
                int i = start_row + d;
                int j = diag - i;

                if (i == 0 || j == 0) {
                    curr[j] = 0;
                } else {
                    int diag_score = (seqA[i - 1] == seqB[j - 1]) ? MATCH_SCORE : MISMATCH_SCORE;
                        diag_score = prev1[j - 1] + diag_score;
                    int up_score   = prev2[j]     + GAP_SCORE;
                    int left_score = prev2[j - 1] + GAP_SCORE;

                    curr[j] = max({diag_score, up_score, left_score, 0});
                }

                if (curr[j] > local_max) {
                    local_max = curr[j];
                    local_i = i;
                    local_j = j;
                }
            }
            #pragma omp barrier
            #pragma omp critical
            {
                if (local_max > max_score){
                    max_score = local_max;
                    max_i = local_i;
                    max_j = local_j;
                }
            }
            
        }

        prev1.swap(prev2);
        prev2.swap(curr);
        fill(curr.begin(), curr.end(), 0);


    }

}






int main(int argc, char **argv) {

    // ====================
    // generate sequence
    // ====================
    string seqA; // = "GATCTCGT";
    string seqB; // = "GATAGCAT";


    int length = 10000;
    double similarity = 0.7;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);




    //  =====================================================
    //  serial diag
    //  =====================================================
    int max_score_diag = 0, max_i_diag = 0, max_j_diag = 0;

    auto start_serial_diag = chrono::high_resolution_clock::now();
    SmithWaterman_serial_diag(seqA, seqB, max_score_diag, max_i_diag, max_j_diag);
    auto end_serial_diag = chrono::high_resolution_clock::now();

    auto duration_serial_diag = chrono::duration_cast<chrono::nanoseconds>(end_serial_diag - start_serial_diag);


    //  =====================================================
    //  serial table
    //  =====================================================
    int maxScore_serial_row = 0, maxi_serial_row = 0, maxj_serial_row = 0;

    vector<vector<int>> serial_score(seqA.size() + 1, vector<int>(seqB.size() + 1, 0));
    auto start_serial_row = chrono::high_resolution_clock::now();
	SmithWaterman_serial_table(serial_score,seqA,seqB, maxScore_serial_row, maxi_serial_row, maxj_serial_row);
    auto end_serial_row = chrono::high_resolution_clock::now();

    auto duration_serial_row = chrono::duration_cast<chrono::nanoseconds>(end_serial_row - start_serial_row);

    //  =====================================================
    //  SIMD version
    //  =====================================================
    int maxScore_parallel_diag = 0, maxi_parallel_daig = 0, maxj_parallel_diag = 0;
    reverse(seqB.begin(), seqB.end());
    auto start_parallel_diag = chrono::high_resolution_clock::now();
    SmithWaterman_parallel_diag(seqA, seqB, maxScore_parallel_diag, maxi_parallel_daig, maxj_parallel_diag);
    auto end_parallel_diag = chrono::high_resolution_clock::now();
    auto duration_parallel_diag = chrono::duration_cast<chrono::nanoseconds>(end_parallel_diag - start_parallel_diag);

    
    // Accuracy test of diag
    if (ans_accuracy(maxScore_serial_row, max_score_diag, maxi_serial_row, max_i_diag, maxj_serial_row, max_j_diag))
        cout << "| diag score_test passed |" << endl;
    else
        cout << "| diag score_test failed |" << endl;
    
    // Accuracy test of SIMD
    if (ans_accuracy(maxScore_serial_row, maxScore_parallel_diag, maxi_serial_row, maxi_parallel_daig, maxj_serial_row, maxj_parallel_diag))
        cout << "| SIMD score_test passed |" << endl << endl;
    else
        cout << "| SIMD score_test failed |" << endl << endl;

    // Output max scores and execution times
    cout << "Row_serial Score    : " << maxScore_serial_row << " at (" << maxi_serial_row << ", " << maxj_serial_row << ")" << endl;
    cout << "Diag_serial Score   : " << max_score_diag << " at (" << max_i_diag << ", " << max_j_diag << ")" << endl;
    cout << "Diag_parallel Score : " << maxScore_parallel_diag << " at (" << maxi_parallel_daig << ", " << maxj_parallel_diag << ")" << endl << endl;

    cout << "==========   Row_serial  ==========" << endl;
    cout << "time: " << duration_serial_row.count() << " ns" << endl;
    cout << "==========  Diag_serial  ==========" << endl;
    cout << "time: " << duration_serial_diag.count() << " ns" << endl;
    cout << "========== Diag_parallel ==========" << endl;
    cout << "time: " << duration_parallel_diag.count() << " ns" << endl << endl;

    //cout << "Speedup of Diag_serial v.s. Row_serial: " << (double)duration_serial_row.count() / (double)duration_serial_diag.count() << endl;
    cout << "Speedup of Diag_parallel v.s. Row_serial: " << (double)duration_serial_row.count() / (double)duration_parallel_diag.count() << endl;
    cout << "Speedup of Diag_parallel v.s. Diag_serial : " << (double)duration_serial_diag.count() / (double)duration_parallel_diag.count() << endl;

    return 0;
}
/*
make; srun ./sw
*/