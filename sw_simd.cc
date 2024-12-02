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

#define BEFORE_MAX 0
#define JUST_AFTER 1
#define ATFER_MAX 2

using namespace std;

int GAP_SCORE = -7;
int MATCH_SCORE = 5;



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



void SmithWaterman_serial(vector<vector<int>>& score,const string seqA,const string seqB, int &max_score_serial, int &max_i_serial, int &max_j_serial){
	int mismatch_score;
    for(int i=1; i<score.size();i++){
		for (int j=1;j<score[i].size();j++){
			 if (seqA[i-1]==seqB[j-1]){
				score[i][j]=score[i-1][j-1] + MATCH_SCORE;
			}else {
				if ((seqA[i - 1] == 'A' && seqB[j - 1] == 'G') || (seqA[i - 1] == 'G' && seqB[j - 1] == 'A') ||
                    (seqA[i - 1] == 'C' && seqB[j - 1] == 'T') || (seqA[i - 1] == 'T' && seqB[j - 1] == 'C')) {
                        mismatch_score = -1;
                }
                else
                {
                    mismatch_score = -3;
                }
				score[i][j] = max({score[i][j-1]-GAP_SCORE,score[i-1][j]-GAP_SCORE,score[i-1][j-1]+mismatch_score,0});
			}
			if (score[i][j]>max_score_serial){
				max_i_serial=i;
				max_j_serial=j;
				max_score_serial=score[i][j];	
			}
		}
	}
}




void SmithWaterman_parallel(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j) {
    const int MATCH_SCORE = 2;
    const int GAP_SCORE = -2;
    const int MISMATCH_SCORE = -3;

    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    vector<int> prev1(rows, 0);
    vector<int> prev2(rows, 0);
    vector<int> curr(rows, 0);

    for (int diag = 2; diag < rows + cols - 1; ++diag) {
        int start_row = max(0, diag - (rows - 1));
        int end_row = min(diag, rows - 1);
        int len_diag = end_row - start_row + 1;

        int state = (start_row == 0) ? BEFORE_MAX : (start_row == 1) ? MAX_DIAG : AFTER_MAX;


        __m256i vec_match_score    = _mm256_set1_epi8(MATCH_SCORE);
        __m256i vec_mismatch_score = _mm256_set1_epi8(MISMATCH_SCORE);
        __m256i vec_gap_score      = _mm256_set1_epi8(GAP_SCORE);

        int d = 0;
        for (; d + 8 <= len_diag; d += 8) {
            
            __m256i vec_seqA = _mm256_loadu_si256((__m256i *)(seqA.data() + start_row + d - 1));
            __m256i vec_seqB = _mm256_loadu_si256((__m256i *)(seqB.data() + diag - (start_row + d) - 1));

            __m256i vec_match    = _mm256_cmpeq_epi8(vec_seqA, vec_seqB);
            __m256i vec_mismatch = _mm256_set1_epi8(1);
            
            __m256i vec_score = _mm256_blendv_epi8(vec_mismatch_score, vec_match_score, vec_match);
            
            __m256i vec_prev1, vec_prev2;
            if (state == BEFORE_MAX) {
                vec_prev1 = _mm256_loadu_si256((__m256i *)(prev1.data() + d - 1));
                vec_prev2 = _mm256_max_epi32(_mm256_loadu_si256((__m256i *)(prev2.data() + d - 1)),
                                             _mm256_loadu_si256((__m256i *)(prev2.data() +   d  )));

            } else if (state == MAX_DIAG) {
                vec_prev1 = _mm256_loadu_si256((__m256i *)(prev1.data() + d));
                vec_prev2 = _mm256_max_epi32(_mm256_loadu_si256((__m256i *)(prev2.data() + d - 1)),
                                             _mm256_loadu_si256((__m256i *)(prev2.data() +   d  )));

            } else { 
                vec_prev1 = _mm256_loadu_si256((__m256i *)(prev1.data() + d + 1));
                vec_prev2 = _mm256_max_epi32(_mm256_loadu_si256((__m256i *)(prev2.data() + d + 1)),
                                             _mm256_loadu_si256((__m256i *)(prev2.data() +   d  )));
            }

            
            
            __m256i vec_curr = _mm256_max_epi8(_mm256_add_epi8(vec_prev1, vec_score), vec_gap_score);

            
            _mm256_storeu_si256((__m256i *)(curr.data() + d), vec_curr);

            
            for (int k = 0; k < 8; ++k) {
                int score = curr[d + k];
                if (score > max_score) {
                    max_score = score;
                    max_i = start_row + d + k;
                    max_j = diag - max_i;
                }
            }
        }

        
        for (; d < len_diag; ++d) {
            int i = start_row + d;
            int j = diag - i;
            if (i == 0 || j == 0) {
                curr[d] = 0;
            } else {
                int score = (seqA[i - 1] == seqB[j - 1]) ? MATCH_SCORE : MISMATCH_SCORE;
                if (state == BEFORE_MAX) {
                    curr[d] = max(prev1[d - 1] + score, max(prev2[d - 1] + GAP_SCORE, 0));
                } else if (state == MAX_DIAG) {
                    curr[d] = max(prev1[d] + score, max(prev2[d] + GAP_SCORE, 0));
                } else {
                    curr[d] = max(prev1[d + 1] + score, max(prev2[d + 1] + GAP_SCORE, 0));
                }
            }

            if (curr[d] > max_score) {
                max_score = curr[d];
                max_i = i;
                max_j = j;
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


    int length = 50000;
    double similarity = 0.7;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);


    //  =====================================================
    //  parallel version of wavefront with two anti-diagonals
    //  =====================================================
    int max_score_parallel = 0, max_i_parallel = 0, max_j_parallel = 0;

    auto start_parallel = chrono::high_resolution_clock::now();
    SmithWaterman_SIMD(seqA, seqB, max_score_parallel, max_i_parallel, max_j_parallel);
    auto end_parallel = chrono::high_resolution_clock::now();
    auto duration_parallel = chrono::duration_cast<chrono::nanoseconds>(end_parallel - start_parallel);


    //  =====================================================
    //  serial version for accuracy
    //  =====================================================
    int max_score_serial = 0, max_i_serial = 0, max_j_serial = 0;


    std::vector<std::vector<int>> score(seqA.size() + 1, std::vector<int>(seqB.size() + 1, 0));
    auto start_serial = chrono::high_resolution_clock::now();
    //SmithWaterman_serial(seqA, seqB, max_score_serial, max_i_serial, max_j_serial);
	SmithWaterman_serial(score,seqA,seqB, max_score_serial, max_i_serial, max_j_serial);
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::nanoseconds>(end_serial - start_serial);

    
    bool score_test = ans_accuracy(max_score_serial, max_score_parallel, max_i_serial, max_i_parallel, max_j_serial, max_j_parallel);
    if(score_test == true) cout << "| score_test passed |" << endl << endl;
    else cout << "| score_test failed |" << endl << endl;


    cout << "Serial Score   : " << max_score_serial << " at (" << max_i_serial << ", " << max_j_serial << ")" << endl;
    cout << "Parallel Score : " << max_score_parallel << " at (" << max_i_parallel << ", " << max_j_parallel << ")" << endl << endl;


    cout << "==========  serial  ==========" << endl;
    cout << "time: " << duration_serial.count() << " ns" << endl;
    cout << "========== parallel ==========" << endl;
    cout << "time: " << duration_parallel.count() << " ns" << endl << endl;


    cout << "speedup: " << (double)duration_serial.count() / (double)duration_parallel.count() << endl;

    return 0;
}
/*
make; srun -c 1 ./sw
*/