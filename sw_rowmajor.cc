#include <thread>
#include <ctime>
#include <cstdlib>
#include <random>
#include <iostream>
#include <vector>
#include <mutex>
#include <string>
#include <algorithm> 
#include <chrono>

using namespace std;

#define gap_score 7
#define match_score 5
#define mismatch_score -1


static inline void calculate(const string &seqA, const string &seqB, vector<vector<int>> &score, int &max_score, int &max_i, int &max_j, int i, int j){

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

    if (score[i][j] > max_score) {
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
            calculate(seqA, seqB, score, max_score, max_i, max_j, i, j);
        }
    }
}


// void SmithWaterman_wavefront(const string &seqA, const string &seqB, 
//                              vector<vector<int>> &score, int &max_score, int &max_i, int &max_j) {
//     int rows = seqA.size() + 1;
//     int cols = seqB.size() + 1;


//     score.assign(rows, vector<int>(cols, 0));

  
//     mutex max_mutex;

//     vector<thread> threads;

    
//     auto worker = [&](int start_diag, int end_diag) {
//         int local_max_score = 0;
//         int local_max_i = 0, local_max_j = 0;

//         for (int diag = start_diag; diag < end_diag; ++diag) {
//             for (int i = max(1, diag - cols + 1); i < min(rows, diag); ++i) {
//                 int j = diag - i;
//                 calculate(seqA, seqB, score, local_max_score, local_max_i, local_max_j, i, j);
//             }
//         }

        
//         lock_guard<mutex> lock(max_mutex);
//         if (local_max_score > max_score) {
//             max_score = local_max_score;
//             max_i = local_max_i;
//             max_j = local_max_j;
//         }
//     };
// }

void SmithWaterman_wavefront(const string &seqA, const string &seqB, 
                             vector<vector<int>> &score, int &max_score, int &max_i, int &max_j) {
    int rows = seqA.size() + 1;
    int cols = seqB.size() + 1;

    // 初始化矩陣
    score.assign(rows, vector<int>(cols, 0));

    // 鎖，用於保護全局最大值
    mutex max_mutex;

    // 獲取執行緒數
    int threadNum = thread::hardware_concurrency();
    vector<thread> threads;

    // 按「水平對角線」分配工作
    for (int diag = 1; diag <= rows + cols - 2; ++diag) {
        // 計算該對角線上所有點的範圍
        vector<pair<int, int>> tasks; // 儲存對角線上的所有座標 (i, j)

        for (int i = max(1, diag - cols + 1); i < min(rows, diag + 1); ++i) {
            int j = diag - i;
            tasks.emplace_back(i, j);
        }

        // 平均分配給每個執行緒
        auto worker = [&](int start, int end, int &local_max_score, int &local_max_i, int &local_max_j) {
            for (int t = start; t < end; ++t) {
                int i = tasks[t].first;
                int j = tasks[t].second;
                calculate(seqA, seqB, score, local_max_score, local_max_i, local_max_j, i, j);
            }
        };

        // 分配任務給執行緒
        int taskPerThread = (tasks.size() + threadNum - 1) / threadNum;
        vector<int> local_max_scores(threadNum, 0);
        vector<int> local_max_is(threadNum, 0);
        vector<int> local_max_js(threadNum, 0);

        threads.clear();
        for (int t = 0; t < threadNum; ++t) {
            int start = t * taskPerThread;
            int end = min((t + 1) * taskPerThread, (int)tasks.size());

            if (start < end) { // 確保任務範圍有效
                threads.emplace_back(worker, start, end, ref(local_max_scores[t]), 
                                     ref(local_max_is[t]), ref(local_max_js[t]));
            }
        }

        // 等待所有執行緒完成
        for (auto &t : threads) {
            t.join();
        }

        // 更新全局最大值
        for (int t = 0; t < threadNum; ++t) {
            lock_guard<mutex> lock(max_mutex);
            if (local_max_scores[t] > max_score) {
                max_score = local_max_scores[t];
                max_i = local_max_is[t];
                max_j = local_max_js[t];
            }
        }
    }
}



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
    uniform_int_distribution<> dis(0, 3);
    uniform_real_distribution<> changeChance(0.0, 1.0);

    seqB.clear();
    for (int i = 0; i < length; ++i) {
        if (changeChance(gen) > similarity) {
            seqB.push_back(bases[dis(gen)]);
        } else {
            seqB.push_back(seqA[i]);
        }
    }
}

int main(int argc, char **argv) {
    int length = 10000;
    double similarity = 0.5;

    string seqA, seqB;
    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);



    vector<vector<int>> score(length + 1, vector<int>(length + 1, 0));

    int max_score = 0, max_i = 0, max_j = 0;

    auto start_wavefront = chrono::high_resolution_clock::now();
    SmithWaterman_wavefront(seqA, seqB, score, max_score, max_i, max_j);
    auto end_wavefront = chrono::high_resolution_clock::now();
    auto duration_wavefront = chrono::duration_cast<chrono::nanoseconds>(end_wavefront - start_wavefront);


    max_score = 0, max_i = 0, max_j = 0;

    auto start_serial = chrono::high_resolution_clock::now();
    SmithWaterman_serial(seqA, seqB, score, max_score, max_i, max_j);
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::nanoseconds>(end_serial - start_serial);


    // cout << "=====sequence A=====" << endl << seqA << endl;
    // cout << "====================" << endl << endl;
    // cout << "=====sequence B=====" << endl << seqB << endl;
    // cout << "====================" << endl << endl;

    cout << "=====  serial  =====" << endl;
    cout << "time: " << duration_serial.count() << " ns" << endl;
    cout << "===== wavefront =====" << endl;
    cout << "time: " << duration_wavefront.count() << " ns" << endl;
    cout << "====================" << endl;
    cout << "speedup: " << (double)duration_serial.count() / (double)duration_wavefront.count() << endl;
    cout << "Max Score: " << max_score << " at (" << max_i << ", " << max_j << ")" << endl;

    return 0;
}
/*
make; srun ./sw


*/