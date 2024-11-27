#include <ctime>
#include <omp.h>
#include <cstdlib>
#include <random>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> 
#include <chrono>

using namespace std;

#define gap_score 7
#define match_score 5
#define mismatch_score -1




void generate_random_seq(string &seqA, int lenA) {
    const char bases[] = {'A', 'T', 'C', 'G'};
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> normal_dis(0, 3);

    seqA.clear();
    for (int i = 0; i < lenA; ++i) {
        seqA.push_back(bases[normal_dis(gen)]);
    }
}

void generate_similar_seq(const string &seqA, string &seqB, int length, double similarity) {
    const char bases[] = {'A', 'T', 'C', 'G'};
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> normal_dis(0, 3);
    uniform_real_distribution<> change(0.0, 1.0);

    seqB.clear();
    for (int i = 0; i < length; ++i) 
    {
        if(i < length && change(gen) < similarity) seqB.push_back(seqA[i]);
        else seqB.push_back(bases[normal_dis(gen)]);
    }
}

bool ans_accuracy(int &score_s, int &score_w, int &i_s, int &i_w, int &j_s, int j_w)
{
    return (score_s == score_w && i_s == i_w && j_s == j_w);
}


void calulate(const string &seqA, const string &seqB, vector<vector<int>> &score, int &max_score, int &max_i, int &max_j, int i, int j){

    int mismatch = mismatch_score;

    if (seqA[i - 1] == seqB[j - 1]) {
        score[i][j] = score[i - 1][j - 1] + match_score;
    } else {
        
        if ((seqA[i - 1] == 'A' && seqB[j - 1] == 'G') || (seqA[i - 1] == 'G' && seqB[j - 1] == 'A') ||
            (seqA[i - 1] == 'C' && seqB[j - 1] == 'T') || (seqA[i - 1] == 'T' && seqB[j - 1] == 'C')) {
            mismatch = -1;
        } else {
            mismatch = -3;
        }

        score[i][j] = max({score[i][j - 1] - gap_score, score[i - 1][j] - gap_score, score[i - 1][j - 1] + mismatch, 0});
    }

    if (score[i][j] > max_score) 
    {
        max_score = score[i][j];
        max_i = i;
        max_j = j;
    }
}

void SmithWaterman_serial(const string &seqA, const string &seqB, vector<vector<int>> &score, int &max_score, int &max_i, int &max_j) {
    
    int rows = seqA.size() + 1; 
    int cols = seqB.size() + 1;
    
    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            calulate(seqA, seqB, score, max_score, max_i, max_j, i, j);
        }
    }
}


void SmithWaterman_parallel(const string &seqA, const string &seqB, vector<vector<int>> &score, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;    

    #pragma omp parallel
    {
        int local_max_score = 0;
        int local_i = 0, local_j = 0;

        for (int diag = 2; diag < rows + cols; ++diag)
        {
            #pragma omp for 
            for (int i = max(1, diag - cols + 1); i < min(rows, diag); ++i) 
            {
                int j = diag - i;
                calulate(seqA, seqB, score, local_max_score, local_i, local_j, i, j);
            }
        }

        #pragma omp critical
        {
            if (local_max_score > max_score) 
            {
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
    string seqA;
    string seqB;
    int length = 10000;
    double similarity = 0.7;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);

    // cout << seqA << endl;
    // cout << seqB <<endl;

    vector<vector<int>> score_serial(seqA.size() + 1, vector<int>(seqB.size() + 1, 0));
    vector<vector<int>> score_parallel(seqA.size() + 1, vector<int>(seqB.size() + 1, 0));


    int max_score_serial = 0, max_i_serial = 0, max_j_serial = 0;
    int max_score_parallel = 0, max_i_parallel = 0, max_j_parallel = 0;


    auto start_serial = chrono::high_resolution_clock::now();
    SmithWaterman_serial(seqA, seqB, score_serial, max_score_serial, max_i_serial, max_j_serial);
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::nanoseconds>(end_serial - start_serial);


    auto start_parallel = chrono::high_resolution_clock::now();
    SmithWaterman_parallel(seqA, seqB, score_parallel, max_score_parallel, max_i_parallel, max_j_parallel);
    auto end_parallel = chrono::high_resolution_clock::now();
    auto duration_parallel = chrono::duration_cast<chrono::nanoseconds>(end_parallel - start_parallel);


    bool score_test = ans_accuracy(max_score_serial, max_score_parallel, max_i_serial, max_i_parallel, max_j_serial, max_j_parallel);
    cout << endl;
    if(score_test == true) cout << "| score_test passed |" << endl << endl;
    else cout << "| score_test failed |" << endl << endl;

    cout << "=====  serial  =====" << endl;
    cout << "time: " << duration_serial.count() << " ns" << endl;
    cout << "===== parallel =====" << endl;
    cout << "time: " << duration_parallel.count() << " ns" << endl << endl;
    cout << "speedup: " << (double)duration_serial.count() / (double)duration_parallel.count() << endl;
    cout << "Serial Score: " << max_score_serial << " at (" << max_i_serial << ", " << max_j_serial << ")" << endl;
    cout << "Parallel Score: " << max_score_parallel << " at (" << max_i_serial << ", " << max_j_serial << ")" << endl;
    return 0;
}
/*
make; srun ./sw
*/