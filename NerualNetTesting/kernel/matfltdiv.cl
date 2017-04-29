__kernel void matfltdiv(__global float* a, __const float b, global float* out){
	int i = get_global_id(0);
	
	out[i] = a[i] / b;
}