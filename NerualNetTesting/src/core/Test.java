/*
 * JOCL - Java bindings for OpenCL
 * 
 * Copyright 2009 Marco Hutter - http://www.jocl.org/
 */

package core;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clEnqueueWriteBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clSetKernelArg;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

public class Test {
	
	static cl_context context;
	static cl_command_queue commandQueue;
	
	static cl_program program;
	static cl_kernel kernel;
	
	static long[] global = new long[2];
	static long[] local = new long[2];
	
//	static {
//
//		
//		CL.setExceptionsEnabled(true);
//		
//		final int platformIndex = 0;
//        final long deviceType = CL_DEVICE_TYPE_GPU;
//        final int deviceIndex = 0;
//
//        // Enable exceptions and subsequently omit error checks in this sample
//        CL.setExceptionsEnabled(true);
//
//        // Obtain the number of platforms
//        int numPlatformsArray[] = new int[1];
//        clGetPlatformIDs(0, null, numPlatformsArray);
//        int numPlatforms = numPlatformsArray[0];
//
//        // Obtain a platform ID
//        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
//        clGetPlatformIDs(platforms.length, platforms, null);
//        cl_platform_id platform = platforms[platformIndex];
//
//        // Initialize the context properties
//        cl_context_properties contextProperties = new cl_context_properties();
//        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
//        
//        // Obtain the number of devices for the platform
//        int numDevicesArray[] = new int[1];
//        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
//        int numDevices = numDevicesArray[0];
//        
//        // Obtain a device ID 
//        cl_device_id devices[] = new cl_device_id[numDevices];
//        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
//        cl_device_id device = devices[deviceIndex];
//
//        // Create a context for the selected device
//        context = clCreateContext(
//            contextProperties, 1, new cl_device_id[]{device}, 
//            null, null, null);
//
//		
//	
//        // Create a command-queue for the selected device
//        commandQueue = 
//            clCreateCommandQueue(context, device, 0, null);
//	}
	
	public static void main(String[] args){
		
		Matrix a = new Matrix(10000, 100);
		Matrix b = new Matrix(100, 10000);
		
		a.set(0, 0, 2);
		a.set(1, 1, 9);
		a.set(2, 2, 1);
		
		b.set(0, 0, 3);
		b.set(1, 1, 1);
		b.set(2, 2, -5);
		
		Matrix c0 = Matrix.dot(a, b);
		
		System.out.println(c0.toString());
	}
	
	public static Matrix dot(Matrix a, Matrix b){
		
		int mdim, ndim, pdim;
		
		mdim = a.getRows();
		ndim = b.getColumns();
		pdim = a.getColumns();
		
		int szA, szB, szC;
		
		szA = ndim * pdim;
		szB = pdim * mdim;
		szC = ndim * mdim;
		
		float[] A, B, C;
		
		A = new float[szA];
		B = new float[szB];
		C = new float[szC];
		
		int i = 0;
		
		for (int row = 0; row < a.getRows(); row++)
		{
			for (int col = 0; col < a.getColumns(); col++)
			{
				A[i++] = (float)a.get(row, col);
			}
		}
		
		i = 0;
		
		for (int row = 0; row < b.getRows(); row++)
		{
			for (int col = 0; col < b.getColumns(); col++)
			{
				B[i++] = (float)b.get(row, col);
			}
		}
		
		Pointer pA = Pointer.to(A);
		Pointer pB = Pointer.to(B);
		Pointer pC = Pointer.to(C);
		
		cl_mem aIn, bIn, cOut;
		
		aIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szA, pA, null);
		bIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szB, pB, null);
		cOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szC, pC, null);
		
		clEnqueueWriteBuffer(commandQueue, aIn, CL_TRUE, 0, Sizeof.cl_float * szA, pA, 0, null, null);
		clEnqueueWriteBuffer(commandQueue, bIn, CL_TRUE, 0, Sizeof.cl_float * szB, pB, 0, null, null);
		
		String code = loadCLCode("kernel/mat_mul.cl");
		
		program = clCreateProgramWithSource(context, 1, new String[]{code}, null, null);
		 clBuildProgram(program, 0, null, null, null, null);
		
		kernel = clCreateKernel(program, "matmul", null);
		
		clSetKernelArg(kernel, 0, Sizeof.cl_int, Pointer.to(new int[]{mdim}));
		clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[]{ndim}));
		clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{pdim}));
		clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(new cl_mem[]{aIn}));
		clSetKernelArg(kernel, 4, Sizeof.cl_mem, Pointer.to(new cl_mem[]{bIn}));
		clSetKernelArg(kernel, 5, Sizeof.cl_mem, Pointer.to(new cl_mem[]{cOut}));
		
		global[0] = ndim;
		global[1] = mdim;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, kernel, 2, null, global, null, 0, null, null);
		clEnqueueReadBuffer(commandQueue, cOut, CL_TRUE, 0, Sizeof.cl_float * szC, Pointer.to(C), 0, null, null);
		
		Matrix c = new Matrix(mdim, ndim);
		
		i = 0;
		
		for (int row = 0; row < c.getRows(); row++)
		{
			for (int col = 0; col < c.getColumns(); col++)
			{
				c.set(row, col, C[i++]);
			}
		}
		
		clReleaseMemObject(aIn);
		clReleaseMemObject(bIn);
		clReleaseMemObject(cOut);

		return c;
		
	}
	
	private static String loadCLCode(String fileName) {
		
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
			StringBuffer sb = new StringBuffer();
			String line = null;
			while (true) {
				line = br.readLine();
				if (line == null) {
					break;
				}
				sb.append(line).append("\n");
			}
			return sb.toString();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
			return null;
		}
		
	}
}