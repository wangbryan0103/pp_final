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
#define ATFER_MAX 2
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

bool ans_accuracy(int &score_s, int &score_w, int &i_s, int &i_w, int &j_s, int j_w)
{
    return (score_s == score_w && i_s == i_w && j_s == j_w);
}



void SmithWaterman_serial(vector<vector<int>>& score,const string seq1,const string seq2, int &max_score_serial, int &max_i_serial, int &max_j_serial){
	int mismatch_score;
    for(int i=1; i<score.size();i++){
		for (int j=1;j<score[i].size();j++){
			 if (seq1[i-1]==seq2[j-1]){
				score[i][j]=score[i-1][j-1] + match_score;
			}else {
				if(seq1[i-1]=='A'&&seq2[j-1]=='G'||seq1[i-1]=='G'&&seq2[j-1]=='A'||seq1[i-1]=='C'&&seq2[j-1]=='T'||seq1[i-1]=='T'&&seq2[j-1]=='C') mismatch_score=-1;
				else mismatch_score=-3;
				score[i][j] = max({score[i][j-1]-gap_score,score[i-1][j]-gap_score,score[i-1][j-1]+mismatch_score,0});
			}
			if (score[i][j]>max_score_serial){
				max_i_serial=i;
				max_j_serial=j;
				max_score_serial=score[i][j];	
			}
		}
	}
}


int calculate_parallel(const string &seqA, const string &seqB, vector<int> &prev1, vector<int> &prev2,int i, int j, int d, int state){

    int mismatch_score;
    int curr;

    if (seqA[i - 1] == seqB[j - 1])
    {
        int index = (state == BEFORE_MAX)? (d-1) : (state == MAX_DIAG)? d : (d+1);
        curr = prev1[index] + match_score;
    }else
    {
        if ((seqA[i - 1] == 'A' && seqB[j - 1] == 'G') || (seqA[i - 1] == 'G' && seqB[j - 1] == 'A') ||
            (seqA[i - 1] == 'C' && seqB[j - 1] == 'T') || (seqA[i - 1] == 'T' && seqB[j - 1] == 'C')) {
            mismatch_score = -1;
        }else
        {
            mismatch_score = -3;
        }
        if(state == BEFORE_MAX){
            curr = max(prev2[d-1] + gap_score, max(prev2[ d ] + gap_score, max(prev1[d-1] + mismatch_score, 0)));
        }else if (state == MAX_DIAG){
            curr = max(prev2[ d ] + gap_score, max(prev2[d-1] + gap_score, max(prev1[ d ] + mismatch_score, 0)));
        }else{
            curr = max(prev2[ d ] + gap_score, max(prev2[d+1] + gap_score, max(prev1[d+1] + mismatch_score, 0)));
        }
    }
    return curr;
}


//assune len_A > len_B
void SmithWaterman_parallel(const string &seqA, const string &seqB, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    vector<int> diag1(rows, 0);
    vector<int> diag2(rows, 0);
    vector<int> diag3(rows, 0);
    vector<int> *prev1 = &diag1;
    vector<int> *prev2 = &diag2;
    vector<int> *curr = &diag3;



    #pragma omp parallel
    {        
        int local_max_score = 0, local_i = 0, local_j = 0;

        for (int diag = 2; diag < rows + cols - 1; ++diag)
        {
            int start_row = max(0, diag - (rows-1)); 
            int end_row = min(diag, rows - 1);
            int len_diag = end_row - start_row + 1;

            int state = (start_row == 0)? 0 : (start_row == 1)? 1 : 2;
            
            #pragma omp for
            for(int d = 0; d < len_diag; ++d)
            {
                int i = start_row + d;
                int j = diag - i;
                if(i == 0 || j == 0) (*curr)[d] = 0;
                else (*curr)[d] = calculate_parallel(seqA, seqB, *prev1, *prev2, i, j, d, state);

                if((*curr)[d] > local_max_score)
                {
                    local_max_score = (*curr)[d];
                    local_i = i;
                    local_j = j;
                }
            }
            #pragma omp barrier 
            
            #pragma omp single
            {
                prev1 = &diag2;
                prev2 = &diag3;
                curr = &diag1;
                fill(diag1.begin(), diag1.end(), 0);
            }
        }

        #pragma omp critical
        {
            if (local_max_score > max_score){
                max_score = local_max_score;
                max_i = local_i;
                max_j = local_j;
            }
        }
    }
}






int main(int argc, char **argv) {

    // ====================
    // generate sequence
    // ====================
    string seqA;// = "GATCTCGT";
    string seqB;// = "GATAGCAT";


    int length = 20000;
    double similarity = 0.7;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);


    //  =====================================================
    //  parallel version of wavefront with two anti-diagonals
    //  =====================================================
    int max_score_parallel = 0, max_i_parallel = 0, max_j_parallel = 0;

    auto start_parallel = chrono::high_resolution_clock::now();
    SmithWaterman_parallel(seqA, seqB, max_score_parallel, max_i_parallel, max_j_parallel);
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
make; srun ./sw
*/