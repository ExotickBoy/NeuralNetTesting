__kernel void matsub(__global float* a, __global float* b, global float* out){
	int i = get_global_id(0);
	
	out[i] = a[i] - b[i];
}