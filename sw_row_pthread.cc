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
#include <pthread.h>
#include <semaphore.h>



using namespace std;

#define gap_score 7
#define match_score 5
#define mismatch_score -1

int length = 10000;

/*string seqA ="GATAGTATTACTAGTACGTTATTTGCCTGCTGC", seqB ="GATCTCGTCACTACTAATCGTACGTCATGCTGCT";
int m = seqA.length();
int n = seqB.length();
vector<std::vector<int>> score(m + 1, std::vector<int>(n + 1, 0));*/
int max_score_parallel = 0, max_i_parallel = 0, max_j_parallel = 0;
string seqA,seqB;
vector<vector<int>> score(length + 1, vector<int>(length + 1, 0));
//sem_t* sems;

int thread_count;

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
    auto start_ = chrono::high_resolution_clock::now();
    int rows = seqA.size() + 1; 
    int cols = seqB.size() + 1;
    
    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            calculate(seqA, seqB, score, max_score, max_i, max_j, i, j);
        }
    }
    auto end_ = chrono::high_resolution_clock::now();
    auto minThread = chrono::duration_cast<chrono::nanoseconds>(end_ - start_);
    cout<<minThread.count()<<endl;
}
pthread_mutex_t mut1 = PTHREAD_MUTEX_INITIALIZER
pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int current_stage = 0; 
void* threadcal(void* rank){
    auto start_ = chrono::high_resolution_clock::now();
    int rows = seqA.size() + 1; 
    int cols = seqB.size() + 1;

    long my_rank = (long) rank;
    
    int my_s = rows/thread_count * my_rank;
    int my_e = rows/thread_count * (my_rank+1);
    //use row cut
    /*if (my_rank==0){
        
        for (int i = my_s+1; i < (my_e+1); ++i) {
            for (int j = 1; j < cols; ++j) {
                calculate(seqA, seqB, score, max_score_parallel, max_i_parallel, max_j_parallel, i, j);
            }
        }
        pthread_mutex_lock(&mut);
        current_stage++;
        pthread_cond_broadcast(&cond); 
        pthread_mutex_unlock(&mut);
    }else{
        pthread_mutex_lock(&mut);
        while (current_stage != my_rank)  {
            pthread_cond_wait(&cond, &mut);
        }
        pthread_mutex_unlock(&mut);
        for (int i = my_s; i < (my_e); ++i) {
            for (int j = 1; j < cols; ++j) {
                calculate(seqA, seqB, score, max_score_parallel, max_i_parallel, max_j_parallel, i, j);
            }
        }
        pthread_mutex_lock(&mut);
        current_stage++;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mut);
    }*/


    // two threads
    if (my_rank == 0){
        for (int i = 1; i < rows; ++i) {
            calculate(seqA, seqB, score, max_score_parallel, max_i_parallel, max_j_parallel, i, 1);
        }
        for (int j = 2; j < cols; ++j) {
            calculate(seqA, seqB, score, max_score_parallel, max_i_parallel,max_j_parallel, 1, j);
        }
        for (int j = 2; j < cols/2; ++j) {
            calculate(seqA, seqB, score, max_score_parallel,max_i_parallel, max_j_parallel, 2, j);
        }
        for(int i = 3; i < rows; ++i) {
            for (int j = 2; j < cols/2; ++j) {
                calculate(seqA, seqB, score, max_score_parallel, max_i_parallel, max_j_parallel, i, j);
                if(j == cols/2 -1){
                    pthread_mutex_lock(&mut);
                    current_stage = i;
                    pthread_cond_broadcast(&cond);
                    pthread_mutex_unlock(&mut);
                }
            }
        }
    }else{
        for(int i = 2; i < rows; ++i) {
            pthread_mutex_lock(&mut1);
            if (i != rows-1){
                while (current_stage < i+1){
                    pthread_cond_wait(&cond, &mut1);
                }
            }
            pthread_mutex_unlock(&mut1);
            
            for (int j = cols/2-1; j < cols; ++j) {
                calculate(seqA, seqB, score,max_score_parallel, max_i_parallel, max_j_parallel, i, j);
            }
        }
    }
    auto end_ = chrono::high_resolution_clock::now();
    auto minThread = chrono::duration_cast<chrono::nanoseconds>(end_ - start_);
    cout<<minThread.count()<<endl;
    return NULL;
}
bool ans_accuracy(int &score_s, int &score_w, int &i_s, int &i_w, int &j_s, int j_w)
{
    return (score_s == score_w && i_s == i_w && j_s == j_w);
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
int main(int argc, char **argv) {
    double similarity = 0.5;

    generate_random_seq(seqA, length);
    generate_similar_seq(seqA, seqB, length, similarity);
    thread_count = strtol(argv[1], NULL, 10);
    auto strat_rowmajor = chrono::high_resolution_clock::now();

    int thread;
    pthread_t* thread_handles;
    
    //sems = (sem_t*) malloc(thread_count*sizeof(sem_t));
    thread_handles = (pthread_t*) malloc (thread_count*sizeof(pthread_t));
    /*sem_init(&sems[0], 0, 1);
    for (thread = 1; thread < thread_count; thread++)
        sem_init(&sems[thread], 0, 0);*/
    for (thread = 0; thread < thread_count; thread++){
        
        pthread_create(&thread_handles[thread], NULL,threadcal, (void*)thread); 
        
    }
    for (thread = 0; thread < thread_count; thread++){
        pthread_join(thread_handles[thread], NULL);
        //sem_destroy(&sems[thread]);
    }
    pthread_mutex_destroy(&mut);
    free(thread_handles);
    //free(sems);
    auto end_rowmajor = chrono::high_resolution_clock::now();
    auto duration_rowmajor = chrono::duration_cast<chrono::nanoseconds>(end_rowmajor - strat_rowmajor);

    vector<vector<int>> score_serial(length + 1, vector<int>(length + 1, 0));
    int max_score_serial = 0, max_i_serial = 0, max_j_serial = 0;

    auto start_serial = chrono::high_resolution_clock::now();
    SmithWaterman_serial(seqA, seqB, score_serial, max_score_serial, max_i_serial, max_j_serial);
    auto end_serial = chrono::high_resolution_clock::now();
    auto duration_serial = chrono::duration_cast<chrono::nanoseconds>(end_serial - start_serial);

    /*cout<<"   ";
    for(int k=0;k<seqB.length();k++)
        cout<<seqB[k]<<"  ";
	    cout<<endl;
        for(int i=1;i<score.size();i++){
		    cout<<seqA[i-1]<<"  ";
                for (int j=1;j<score[i].size();j++){
			        cout<<score[i][j]<<"  ";
		}
		cout<<endl;
	}*/

    // cout << "=====sequence A=====" << endl << seqA << endl;
    // cout << "====================" << endl << endl;
    // cout << "=====sequence B=====" << endl << seqB << endl;
    // cout << "====================" << endl << endl;
    bool score_test = ans_accuracy(max_score_serial, max_score_parallel, max_i_serial, max_i_parallel, max_j_serial, max_j_parallel);
   

    if(score_test == true) cout << "| score_test passed |" << endl << endl;
    else cout << "| score_test failed |" << endl << endl;

    cout << "=====  serial  =====" << endl;
    cout << "time: " << duration_serial.count() << " ns" << endl;
    cout << "===== rowmajor =====" << endl;
    cout << "time: " << duration_rowmajor.count() << " ns" << endl;
    cout << "====================" << endl;
    cout << "speedup: " << (double)duration_serial.count() / (double)duration_rowmajor.count() << endl;
    cout << "Max Score: " << max_score_parallel << " at (" << max_i_parallel << ", " << max_j_parallel << ")" << endl;

    return 0;
}
