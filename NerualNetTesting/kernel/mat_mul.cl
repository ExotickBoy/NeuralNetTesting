__kernel void matmul(
	const int mdim, const int ndim, const int pdim,
	__global float *A, __global float *B, __global float *C)
{
	int i,j,k;
	i = get_global_id(0);
	j = get_global_id(1);
	
	float tmp = 0.0f;
	
	for (k=0; k<pdim; k++)
 tmp += A[i*ndim+k] * B[k*pdim+j];
 C[i*ndim+j] += tmp; 
}