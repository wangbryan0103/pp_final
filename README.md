# 用Multi-Threads 以及 SIMD 加速運算 Smith-Waterman algorithm
2024 平行化程式設計期末Project 第七組 <br>
主要目標 **加快Smith-Waterman演算法** <br>
testfile 僅為寫code時的測試版可以忽略
## 環境
課堂提供的Server即可
## 使用方法
### Row based pthread
p.s 我們的方法是用2個threads所以不用改數字2
```
make clean
make METHOD=sw_row_pthread
srun ./sw 2
```
### Row based SIMD
```
make clean
make METHOD=sw_row_simd
srun ./sw
```
### Diag based openMP
-c N是threads數量
```
make clean
make METHOD=sw_diag_openmp
srun -c N ./sw
```
### Diag based SIMD
```
make clean
make METHOD=sw_diag_simd
srun ./sw
```
