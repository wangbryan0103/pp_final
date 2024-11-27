#include<iostream>
#include<time.h>
#include<string>
#include<vector>
#include<algorithm>

#define SECOND 1e9
int gappenalty=7;
int match=5;
int mismatch=-1;
int max_i;
int max_j;
std::string ansseq;
bool flag=false;
void smithWatermanBacktrack(const std::string& seq1, const std::string& seq2,const int i,const int j,const std::vector<std::vector<int>>& dp) {
    if (i == 0 || j == 0||dp[i][j]==0) {
	flag=true;
        return;
    }
    if (dp[i][j] == dp[i-1][j-1] + match&& seq1[i-1]==seq2[j-1]&&!flag) {
	ansseq+=seq1[i-1];
        smithWatermanBacktrack(seq1, seq2, i-1, j-1, dp);
    }
    if(seq1[i-1]=='A'&&seq2[j-1]=='G'||seq1[i-1]=='G'&&seq2[j-1]=='A'||seq1[i-1]=='C'&&seq2[j-1]=='T'||seq1[i-1]=='T'&&seq2[j-1]=='C') mismatch=-1;
                                else  if(seq1[i-1]=='C'&&seq2[j-1]=='G'||seq1[i-1]=='G'&&seq2[j-1]=='C'||seq1[i-1]=='T'&&seq2[j-1]=='A'||seq1[i-1]=='A'&&seq2[j-1]=='T'||seq1[i-1]=='T'&&seq2[j-1]=='G'||seq1[i-1]=='G'&&seq2[j-1]=='T'||seq1[i-1]=='A'&&seq2[j-1]=='C'||seq1[i-1]=='C'&&seq2[j-1]=='A') mismatch=-3;
    if (dp[i][j] == dp[i-1][j-1] + mismatch&&!flag){
	    ansseq+="X";
	    smithWatermanBacktrack(seq1,seq2,i-1,j-1,dp);
    }
    if (dp[i][j] == dp[i-1][j] - gappenalty&&!flag) {
	ansseq+="-";
        smithWatermanBacktrack(seq1, seq2, i-1, j, dp);
    }
    if (dp[i][j] == dp[i][j-1] - gappenalty&&!flag) {
	ansseq+="-";
        smithWatermanBacktrack(seq1, seq2, i, j-1, dp);
    }
}

void Smi_water(std::vector<std::vector<int>>& score,const std::string seq1,const std::string seq2){
	int max_tmp = 1;
	for(int i=1;i<score.size();i++){
		for (int j=1;j<score[i].size();j++){
			 if (seq1[i-1]==seq2[j-1]){
				score[i][j]=score[i-1][j-1] + match;
			}else {
				if(seq1[i-1]=='A'&&seq2[j-1]=='G'||seq1[i-1]=='G'&&seq2[j-1]=='A'||seq1[i-1]=='C'&&seq2[j-1]=='T'||seq1[i-1]=='T'&&seq2[j-1]=='C') mismatch=-1;
				else  if(seq1[i-1]=='C'&&seq2[j-1]=='G'||seq1[i-1]=='G'&&seq2[j-1]=='C'||seq1[i-1]=='T'&&seq2[j-1]=='A'||seq1[i-1]=='A'&&seq2[j-1]=='T'||seq1[i-1]=='T'&&seq2[j-1]=='G'||seq1[i-1]=='G'&&seq2[j-1]=='T'||seq1[i-1]=='A'&&seq2[j-1]=='C'||seq1[i-1]=='C'&&seq2[j-1]=='A') mismatch=-3;
				score[i][j] = std::max({score[i][j-1]-gappenalty,score[i-1][j]-gappenalty,score[i-1][j-1]+mismatch,0});
			}
			if (score[i][j]>max_tmp){
				max_i=i;
				max_j=j;
				max_tmp=score[i][j];	
			}
		}
	}
}

int main(){
	struct timespec start = {0, 0};
        struct timespec end = {0, 0};
        clock_gettime(CLOCK_REALTIME, &start);
	
	std::string seq1="GATAGTATTACTAGTACGTTATTTGCCTGCTGC",seq2="GATCTCGTCACTACTAATCGTACGTCATGCTGCT";
	int m = seq1.length();
    	int n = seq2.length();
	std::vector<std::vector<int>> score(m + 1, std::vector<int>(n + 1, 0));
	Smi_water(score,seq1,seq2);
	std::cout<<"   ";
	for(int k=0;k<seq2.length();k++)std::cout<<seq2[k]<<"  ";
	std::cout<<std::endl;
        for(int i=1;i<score.size();i++){
		std::cout<<seq1[i-1]<<"  ";
                for (int j=1;j<score[i].size();j++){
			std::cout<<score[i][j]<<"  ";
		}
		std::cout<<std::endl;
	}
	//std::cout<<max_i<<max_j<<std::endl;
	smithWatermanBacktrack(seq1,seq2,max_i,max_j,score);
	std::reverse(ansseq.begin(),ansseq.end());
	std::cout<<ansseq<<std::endl;
	clock_gettime(CLOCK_REALTIME, &end);
	std::cout<<(double)((end.tv_sec - start.tv_sec) +
                       (end.tv_nsec - start.tv_nsec))/SECOND;
}

