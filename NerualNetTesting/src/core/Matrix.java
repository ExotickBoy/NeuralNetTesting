package core;

import static org.jocl.CL.*;
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
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;

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

@SuppressWarnings("deprecation")
public class Matrix implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private static final NumberFormat FORMATTER = new DecimalFormat("#0.00");
	
	private static cl_context context;
	private static cl_command_queue commandQueue;
	
	private static cl_kernel mulKernel;
	private static cl_kernel sigKernel;
	
	private static long[] global = new long[2];
	private static long[] local = new long[2];
	
	private int rows;
	private int columns;
	
	private float[] data;
	
	static {
		
		CL.setExceptionsEnabled(true);
		
		final int platformIndex = 0;
		final long deviceType = CL_DEVICE_TYPE_ALL;
		final int deviceIndex = 0;
		
		// Enable exceptions and subsequently omit error checks in this sample
		CL.setExceptionsEnabled(true);
		
		// Obtain the number of platforms
		int numPlatformsArray[] = new int[1];
		clGetPlatformIDs(0, null, numPlatformsArray);
		int numPlatforms = numPlatformsArray[0];
		
		// Obtain a platform ID
		cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
		clGetPlatformIDs(platforms.length, platforms, null);
		cl_platform_id platform = platforms[platformIndex];
		
		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
		
		// Obtain the number of devices for the platform
		int numDevicesArray[] = new int[1];
		clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
		int numDevices = numDevicesArray[0];
		
		// Obtain a device ID
		cl_device_id devices[] = new cl_device_id[numDevices];
		clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
		cl_device_id device = devices[deviceIndex];
		
		// Create a context for the selected device
		context = clCreateContext(contextProperties, 1, new cl_device_id[] { device }, null, null, null);
		
		// Create a command-queue for the selected device
		commandQueue = clCreateCommandQueue(context, device, 0, null);
		
		try {
			
			mulKernel = loadKernel(new File("kernel/mat_mul.cl"), "matmul");
			sigKernel = loadKernel(new File("kernel/mat_sig.cl"), "matsig");
			
		} catch (IOException e) {
			
			e.printStackTrace();
			System.exit(1);
			
		}
		
	}
	
	public static cl_kernel loadKernel(File file, String kernelName) throws IOException {
		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
		StringBuilder sb = new StringBuilder();
		String line = null;
		while ((line = br.readLine()) != null) {
			sb.append(line).append("\n");
		}
		br.close();
		String code = sb.toString();
		
		cl_program program = clCreateProgramWithSource(context, 1, new String[] { code }, null, null);
		clBuildProgram(program, 0, null, null, null, null);
		
		cl_kernel kernel = clCreateKernel(program, kernelName, null);
		
		return kernel;
		
	}
	
	public Matrix(int rows, int columns) {
		
		this(rows, columns, new double[rows * columns]);
		
	}
	
	public Matrix(int rows, int columns, float[] data) {
		
		this.data = data;
		
		this.rows = rows;
		this.columns = columns;
		
	}
	
	public Matrix(int rows, int columns, double[] data) {
		
		this.data = new float[data.length];
		
		this.rows = rows;
		this.columns = columns;
		
		for (int i = 0; i < data.length; i++) {
			this.data[i] = (float) data[i];
		}
		
	}
	
	public Matrix(int rows, int columns, Supplier<Double> generator) {
		
		this(rows, columns);
		for (int i = 0; i < rows * columns; i++) {
			data[i] = (float) generator.get().doubleValue();
		}
		
	}
	
	public Matrix(Matrix a) {
		
		this.rows = a.rows;
		this.columns = a.columns;
		this.data = a.data.clone();
		
	}
	
	public int getColumns() {
		
		return columns;
		
	}
	
	public int getRows() {
		
		return rows;
		
	}
	
	public void set(int row, int column, double c) {
		
		data[row * columns + column] = (float) c;
		
	}
	
	public void set(int index, double c) {
		
		data[index] = (float) c;
		
	}
	
	public double get(int row, int column) {
		
		return data[row * columns + column];
		
	}
	
	public double get(int index) {
		
		return data[index];
		
	}
	
	public static Matrix map(Matrix input, Function<Double, Double> function) {
		
		Matrix result = new Matrix(input.rows, input.columns);
		
		IntStream.range(0, input.rows * input.columns).parallel().forEach(i -> {
			
			result.data[i] = (float) function.apply((double) input.data[i]).doubleValue();
			
		});
		
		return result;
		
	}
	
	public static Matrix dot(Matrix a, Matrix b) {
		
		assert a.columns == b.rows;
		
		int mdim, ndim, pdim;
		
		mdim = a.getRows();
		ndim = b.getColumns();
		pdim = a.getColumns();
		
		int szA, szB, szC;
		
		szA = mdim * pdim;
		szB = pdim * ndim;
		szC = ndim * mdim;
		
		float[] A, B, C;
		
		C = new float[szC];
		
		A = a.data;
		B = b.data;
		
		Pointer pA = Pointer.to(A);
		Pointer pB = Pointer.to(B);
		Pointer pC = Pointer.to(C);
		
		cl_mem aIn, bIn, cOut;
		
		aIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szA, pA, null);
		bIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szB, pB, null);
		cOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szC, pC, null);
		
		clEnqueueWriteBuffer(commandQueue, aIn, CL_TRUE, 0, Sizeof.cl_float * szA, pA, 0, null, null);
		clEnqueueWriteBuffer(commandQueue, bIn, CL_TRUE, 0, Sizeof.cl_float * szB, pB, 0, null, null);
		
		clSetKernelArg(mulKernel, 0, Sizeof.cl_int, Pointer.to(new int[] { mdim }));
		clSetKernelArg(mulKernel, 1, Sizeof.cl_int, Pointer.to(new int[] { ndim }));
		clSetKernelArg(mulKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { pdim }));
		clSetKernelArg(mulKernel, 3, Sizeof.cl_mem, Pointer.to(new cl_mem[] { aIn }));
		clSetKernelArg(mulKernel, 4, Sizeof.cl_mem, Pointer.to(new cl_mem[] { bIn }));
		clSetKernelArg(mulKernel, 5, Sizeof.cl_mem, Pointer.to(new cl_mem[] { cOut }));
		
		global[0] = ndim;
		global[1] = mdim;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, mulKernel, 2, null, global, null, 0, null, null);
		clEnqueueReadBuffer(commandQueue, cOut, CL_TRUE, 0, Sizeof.cl_float * szC, Pointer.to(C), 0, null, null);
		
		Matrix c = new Matrix(mdim, ndim, C);
		
		clReleaseMemObject(aIn);
		clReleaseMemObject(bIn);
		clReleaseMemObject(cOut);
		
		return c;
		
	}
	
	public static Matrix multiply(float a, Matrix b) {
		
		Matrix result = new Matrix(b.rows, b.columns);
		
		IntStream.range(0, result.rows * result.columns).parallel().forEach(i -> {
			result.data[i] = a * b.data[i];
		});
		
		return result;
		
	}
	
	public static Matrix multiply(Matrix a, Matrix b) {
		
		assert a.columns == b.columns && a.rows == b.rows;
		
		Matrix result = new Matrix(a.rows, b.columns);
		
		IntStream.range(0, result.rows * result.columns).parallel().forEach(i -> {
			result.data[i] = a.data[i] * b.data[i];
		});
		
		return result;
		
	}
	
	public static Matrix transpose(Matrix a) {
		
		Matrix result = new Matrix(a.columns, a.rows);
		for (int row = 0; row < a.rows; row++) {
			for (int column = 0; column < a.columns; column++) {
				result.data[column * a.rows + row] = a.data[row * a.columns + column];
			}
		}
		
		return result;
	}
	
	public static Matrix add(Matrix a, Matrix b) {
		
		assert a.rows == b.rows && a.columns == b.columns;
		
		Matrix result = new Matrix(a.rows, a.columns);
		IntStream.range(0, result.rows * result.columns).parallel().forEach(i -> {
			result.data[i] = a.data[i] + b.data[i];
		});
		
		return result;
		
	}
	
	public static Matrix sub(Matrix a, Matrix b) {
		
		assert a.rows == b.rows && a.columns == b.columns;
		
		Matrix result = new Matrix(a.rows, a.columns);
		IntStream.range(0, result.rows * result.columns).parallel().forEach(i -> {
			result.data[i] = a.data[i] - b.data[i];
		});
		
		return result;
		
	}
	
	public static double sum(Matrix a) {
		
		return (double) IntStream.range(0, a.rows * a.columns).parallel().mapToDouble(i -> {
			return a.data[i];
		}).sum();
		
	}
	
	public static Matrix sigmoid(Matrix a) {
		
		int rows, columns;
		
		rows = a.getRows();
		columns = a.getColumns();
		
		int szA = rows * columns;
		
		float[] mIn, mOut;
		
		mOut = new float[szA];
		
		mIn = a.data;
		
		Pointer pIn = Pointer.to(mIn);
		Pointer pOut = Pointer.to(mOut);
		
		cl_mem memIn, memOut;
		
		memIn = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szA, pIn, null);
		memOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * szA, pOut, null);
		
		clEnqueueWriteBuffer(commandQueue, memIn, CL_TRUE, 0, Sizeof.cl_float * szA, pIn, 0, null, null);
		
		clSetKernelArg(sigKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { memIn }));
		clSetKernelArg(sigKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { memOut }));
		
		global[0] = rows * columns;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, sigKernel, 1, null, global, null, 0, null, null);
		clEnqueueReadBuffer(commandQueue, memOut, CL_TRUE, 0, Sizeof.cl_float * szA, Pointer.to(mOut), 0, null, null);
		
		Matrix out = new Matrix(rows, columns, mOut);
		
		clReleaseMemObject(memIn);
		clReleaseMemObject(memOut);
		
		return out;
		
	}
	
	@Override
	public String toString() {
		
		StringBuilder string = new StringBuilder();
		
		int i = 0;
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				string.append(FORMATTER.format(data[i++]) + ",\t");
			}
			string.append("\n");
		}
		
		return string.toString();
		
	}
	
}
