#' @title Function \code{gpumult}
#' @description Simple GPU multiplication
#' 
#' @export
#' @return 
#' @param A matrix
#' @param B matrix
#' @useDynLib rgpumult GPUMULT
gpumult = function(A, B) {
  .Call(
    "GPUMULT",
    A,
    B
  )
}
