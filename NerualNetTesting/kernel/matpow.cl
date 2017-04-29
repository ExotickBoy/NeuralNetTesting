__kernel void matpow(__global float* a, __const float b, global float* out){
	int i = get_global_id(0);
	
	out[i] = pow(a[i], b);
}