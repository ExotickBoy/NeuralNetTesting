__kernel void matsig(
	const int rows, const int columns, 
	__global float *mIn, __global float *mOut)
{
	int row,col;
	col = get_global_id(0); // iterates through columns
	row = get_global_id(1); // iterates through rows
	
	mOut[row * columns + col] = 1/(1+exp(-mIn[row * columns + col]);
}