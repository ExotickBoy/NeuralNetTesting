__kernel void fltmul(__const float a, __global float* b, global float* out){
	int i = get_global_id(0);
	
	out[i] = a * b[i];
}