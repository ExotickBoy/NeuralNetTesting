__kernel void matdotbt(
	const int mdim, const int ndim, const int pdim,
	__global float *A, __global float *B, __global float *C)
{
	int row,col,k;
	col = get_global_id(0); // iterates through columns
	row = get_global_id(1); // iterates through rows
	
	float tmp = 0.0f;
	
	for (k=0; k<pdim; k++)
 		tmp += A[row*pdim+k] * B[col*pdim+k];
 	C[row*ndim+col] = tmp;
}