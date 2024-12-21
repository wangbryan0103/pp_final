# 用Multi-Threads 以及 SIMD 加速運算 Smith-Waterman algorithm
2024 平行化程式設計期末Project 第七組 <br>
主要目標 **加快Smith-Waterman演算法**
## 環境
課堂提供的Server即可
## 使用方法
### Row based pthread
```
make clean
make METHOD=sw_row
./sw
```
### Row based SIMD
```
make clean
make METHOD=sw_row_simd
./sw
```
### Dig based openMP
```
make clean
make METHOD=diad_openmp
./sw
```
### Dig based openMP
```
make clean
make METHOD=diad_simd
./sw
```
