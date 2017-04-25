__kernel void matsigprime( 
	__global float *mIn, __global float *mOut)
{
	int i;
	i = get_global_id(0);
	
	float tmp = mIn[i];
	
	mOut[i] = tmp * (1-tmp);
}