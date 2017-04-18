__kernel void matsig( 
	__global float *mIn, __global float *mOut)
{
	int i;
	i = get_global_id(0);
	
	mOut[i] = 1/(1+exp(-mIn[i]));
}