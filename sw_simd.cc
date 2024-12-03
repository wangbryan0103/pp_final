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

#define MATCH_SCORE 3
#define GAP_SCORE -3
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



void SmithWaterman_serial(vector<vector<int>>& score,const string seqA,const string seqB,
                                int &max_score_serial, int &max_i_serial, int &max_j_serial){
	int mismatch_score;
    for(size_t i=1; i<score.size();i++){
		for (size_t j=1;j<score[i].size();j++){
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



void SmithWaterman_SIMD(const std::string &seqA, const std::string &seqB, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    std::vector<int> prev1(rows + 1, 0); 
    std::vector<int> prev2(rows + 1, 0);
    std::vector<int> curr(rows + 1, 0);

    // Convert sequences to int for AVX2 processing, which can only conpute int or floating number
    std::vector<int> seqA_int(seqA.size());
    std::vector<int> seqB_int(seqB.size());

    // data transfer for int array
    for (size_t idx = 0; idx < seqA.size(); ++idx) {
        seqA_int[idx] = (int)seqA[idx];
    }
    for (size_t idx = 0; idx < seqB.size(); ++idx) {
        seqB_int[idx] = (int)seqB[idx];
    }

    // poitner of int array
    int *pSeqAi = seqA_int.data();
    int *pSeqBi = seqB_int.data();


    for (int diag = 2; diag < rows + cols - 1; ++diag) {

        int start_row = std::max(0, diag - (cols - 1));
        int end_row = std::min(diag - 1, rows - 1);
        int len_diag = end_row - start_row + 1;

        int state = (start_row == 0) ? BEFORE_MAX : (start_row == 1) ? NEXT_MAX : AFTER_MAX;

        // vector for distinguish between "normal mismatch" and "serious mismatch"
        __m256i vec_A = _mm256_set1_epi32('A');
        __m256i vec_C = _mm256_set1_epi32('C');
        __m256i vec_G = _mm256_set1_epi32('G');
        __m256i vec_T = _mm256_set1_epi32('T');        

        int d = 0;
        for (; d <= len_diag - 8; d += 7){

            // load the sequence vector (int form)  
            __m256i seqA_chars = _mm256_loadu_si256( (__m256i*)(&pSeqAi[start_row + d]) );
            __m256i seqB_chars = _mm256_loadu_si256( (__m256i*)(&pSeqBi[diag - start_row - d - 7]) );

            // mask that check match or not
            __m256i match_mask = _mm256_cmpeq_epi32(seqA_chars, seqB_chars);

            // ========================================================================================
            // calculate which is "normal mismatch", and "serious mismatch"
            // this part is done by GPT 
            __m256i cmp_seqA_A = _mm256_cmpeq_epi32(seqA_chars, vec_A);
            __m256i cmp_seqA_G = _mm256_cmpeq_epi32(seqA_chars, vec_G);
            __m256i cmp_seqA_C = _mm256_cmpeq_epi32(seqA_chars, vec_C);
            __m256i cmp_seqA_T = _mm256_cmpeq_epi32(seqA_chars, vec_T);

            __m256i cmp_seqB_A = _mm256_cmpeq_epi32(seqB_chars, vec_A);
            __m256i cmp_seqB_G = _mm256_cmpeq_epi32(seqB_chars, vec_G);
            __m256i cmp_seqB_C = _mm256_cmpeq_epi32(seqB_chars, vec_C);
            __m256i cmp_seqB_T = _mm256_cmpeq_epi32(seqB_chars, vec_T);

            __m256i cmp1 = _mm256_and_si256(cmp_seqA_A, cmp_seqB_G);
            __m256i cmp2 = _mm256_and_si256(cmp_seqA_G, cmp_seqB_A);
            __m256i cmp3 = _mm256_and_si256(cmp_seqA_C, cmp_seqB_T);
            __m256i cmp4 = _mm256_and_si256(cmp_seqA_T, cmp_seqB_C);

            __m256i mismatch_mask = _mm256_or_si256(_mm256_or_si256(cmp1, cmp2), _mm256_or_si256(cmp3, cmp4));

            __m256i mismatch_score_vec = _mm256_blendv_epi8(_mm256_set1_epi32(-3), _mm256_set1_epi32(-1), mismatch_mask);
            // ========================================================================================

            // U: upper, L: left 
            __m256i prev1_vec;
            __m256i prev2U_vec;
            __m256i prev2L_vec;
            __m256i match_vec;
            __m256i mismatch_vec;
            __m256i curr_vec;

            // load the vector which have different index in 3 different states
            if(state == BEFORE_MAX){
                prev1_vec  = _mm256_loadu_si256((__m256i*)&prev1[d-1]);
                prev2U_vec = _mm256_loadu_si256((__m256i*)&prev2[ d ]);
                prev2L_vec = _mm256_loadu_si256((__m256i*)&prev2[d-1]);
            }
            else if(state == NEXT_MAX)
            {
                prev1_vec  = _mm256_loadu_si256((__m256i*)&prev1[ d ]);
                prev2U_vec = _mm256_loadu_si256((__m256i*)&prev2[d+1]);
                prev2L_vec = _mm256_loadu_si256((__m256i*)&prev2[ d ]);
            }else
            {
                prev1_vec  = _mm256_loadu_si256((__m256i*)&prev1[d+1]);
                prev2U_vec = _mm256_loadu_si256((__m256i*)&prev2[d+1]);
                prev2L_vec = _mm256_loadu_si256((__m256i*)&prev2[ d ]);
            }
            __m256i t1 = _mm256_add_epi32(prev1_vec , mismatch_score_vec);
            __m256i t2 = _mm256_add_epi32(prev2U_vec, _mm256_set1_epi32(GAP_SCORE));
            __m256i t3 = _mm256_add_epi32(prev2L_vec, _mm256_set1_epi32(GAP_SCORE));

            t1 = _mm256_max_epi32(t1, _mm256_setzero_si256());
            t2 = _mm256_max_epi32(t2, t3);

            mismatch_vec = _mm256_max_epi32(t1, t2);
            match_vec    = _mm256_add_epi32(prev1_vec, _mm256_set1_epi32(MATCH_SCORE));
            
            curr_vec     = _mm256_blendv_epi8(mismatch_vec, match_vec, match_mask);


            // ========== 這裏index不太確定，GPT寫d+1, 我原本寫d ==========
            _mm256_storeu_si256((__m256i*)&curr[d + 1], curr_vec); // Adjusted for padding

            
            for (int k = 0; k < 8; ++k) {
                int score = curr[d + k + 1];
                if (score > max_score) {
                    max_score = score;
                    max_i = start_row + d + k;
                    max_j = diag - max_i;
                }
            }
        }

        
        // Remaining elements
        // just copy the serial version 
        //這裡是讓GPT寫的，應該可以改成一樣平行，然後用mask去擋掉超出範圍的
        for (; d < len_diag; ++d) {
            int i = start_row + d;
            int j = diag - i;
            if (i == 0 || j == 0) {
                //========== 這裡同上，不知道為什麼GPT寫d+1, 沒不確定誰對 ==========
                curr[d + 1] = 0; 
            } else {
                int index_prev = d + 1;
                int mismatch_score;
                if (seqA[i - 1] == seqB[j - 1]) {
                    curr[index_prev] = prev1[index_prev] + MATCH_SCORE;
                } else {
                    if ((seqA[i - 1] == 'A' && seqB[j - 1] == 'G') || (seqA[i - 1] == 'G' && seqB[j - 1] == 'A') ||
                        (seqA[i - 1] == 'C' && seqB[j - 1] == 'T') || (seqA[i - 1] == 'T' && seqB[j - 1] == 'C')) {
                        mismatch_score = -1;
                    } else {
                        mismatch_score = -3;
                    }
                    int t1 = prev2[index_prev - 1] + GAP_SCORE;
                    int t2 = prev2[index_prev] + GAP_SCORE;
                    int t3 = prev1[index_prev] + mismatch_score;
                    curr[index_prev] = max({ t1, t2, t3, 0 });
                }

                if (curr[index_prev] > max_score) {
                    max_score = curr[index_prev];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        // rotate array
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