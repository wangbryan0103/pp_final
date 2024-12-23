# 用Multi-Threads 以及 SIMD 加速運算 Smith-Waterman algorithm
2024 平行化程式設計期末Project 第七組 <br>
主要目標 **加快Smith-Waterman演算法** <br>
testfile 僅為寫code時的測試版可以忽略
## 環境
課堂提供的Server即可
## 使用方法
### Row based pthread
```
make clean
make METHOD=sw_row_pthread
./sw
```
### Row based SIMD
```
make clean
make METHOD=sw_row_simd
./sw
```
### Diag based openMP
```
make clean
make METHOD=sw_diag_openmp
./sw
```
### Diag based SIMD
```
make clean
make METHOD=sw_diag_simd
./sw
```
