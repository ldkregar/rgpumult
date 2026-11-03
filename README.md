# rgpumult

R package with very basic implementation that wraps up CUDA matrix multiplication.
Not tested for memory leaks. 

## Installation 
```r
devtools::install_github("ldkregar/rgpumult")
```

## Usage

```r
library(rgpumult)
library(tictoc)
for(n in c(2000,4000)){
	A=matrix(runif(n*3000),nrow=n,ncol=3000)
	B=matrix(runif(3000*n),nrow=3000,ncol=n)
	tic()
	r=A %*% B
	cat("Results for n=",n,"......\nBase R:");toc()
	tic()
	g=gpumult(A,B)
	cat("\nGPU:");toc()
	cat("\nmax diff=",max(abs(g-r)),"\n")
}
```
