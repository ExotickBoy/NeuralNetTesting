package core;

import static org.jocl.CL.*;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
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
	
	private static cl_context context;
	private static cl_command_queue commandQueue;
	
	private static cl_kernel mulKernel;
	private static cl_kernel sigKernel;
	private static cl_kernel sigPrimeKernel;
	private static cl_kernel dotKernel;
	private static cl_kernel addKernel;
	private static cl_kernel subKernel;
	private static cl_kernel fltMulKernel;
	
	private static long[] global = new long[2];
	private static long[] local = new long[2];
	
	private int rows;
	private int columns;
	
	private int size;
	
	private boolean isTransposed;
	
	private cl_mem mData;
	
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
			
			mulKernel = loadKernel(new File("kernel/matmul.cl"), "matmul");
			dotKernel = loadKernel(new File("kernel/matdot.cl"), "matdot");
			sigKernel = loadKernel(new File("kernel/matsig.cl"), "matsig");
			sigPrimeKernel = loadKernel(new File("kernel/matsigprime.cl"), "matsigprime");
			addKernel = loadKernel(new File("kernel/matadd.cl"), "matadd");
			fltMulKernel = loadKernel(new File("kernel/fltmul.cl"), "fltmul");
			subKernel = loadKernel(new File("kernel/matsub.cl"), "matsub");
			
		} catch (IOException e) {
			
			e.printStackTrace();
			System.exit(1);
			
		}
		
	}
	
	private static cl_kernel loadKernel(File file, String kernelName) throws IOException {
		
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
		
		this(rows, columns, new float[rows * columns]);
		
	}
	
	public Matrix(int rows, int columns, float[] data) {
		
		assert data.length == rows * columns;
		
		this.rows = rows;
		this.columns = columns;
		
		this.size = rows * columns;
		
		this.mData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * size, Pointer.to(data), null);
		
		isTransposed = false;
		
	}
	
	@Deprecated
	public Matrix(int rows, int columns, double[] data) {
		
		this.data = new float[data.length];
		
		this.rows = rows;
		this.columns = columns;
		
		this.size = rows * columns;
		
		for (int i = 0; i < data.length; i++) {
			this.data[i] = (float) data[i];
		}
		
		isTransposed = false;
		
	}
	
	@Deprecated
	public Matrix(int rows, int columns, Supplier<Double> generator) {
		
		this(rows, columns);
		for (int i = 0; i < rows * columns; i++) {
			data[i] = (float) generator.get().doubleValue();
		}
		
	}
	
	public Matrix(Matrix a) {
		
		this.rows = a.rows;
		this.columns = a.columns;
		this.size = rows * columns;
		
		clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * size, Pointer.to(a.mData), null);
		
	}
	
	public float[] getData(){
		
	}
	
	public int getColumns() {
		
		return columns;
		
	}
	
	public int getRows() {
		
		return rows;
		
	}
	
	public int getSize() {
		
		return size;
		
	}
	
	@Deprecated
	public void set(int row, int column, double c) {
		
		data[row * columns + column] = (float) c;
		
	}
	
	@Deprecated
	public void set(int index, double c) {
		
		data[index] = (float) c;
		
	}
	
	@Deprecated
	public double get(int row, int column) {
		
		return data[row * columns + column];
		
	}
	
	@Deprecated
	public double get(int index) {
		
		return data[index];
		
	}
	
	@Deprecated
	public static Matrix map(Matrix input, Function<Double, Double> function) {
		
		Matrix result = new Matrix(input.rows, input.columns);
		
		IntStream.range(0, input.rows * input.columns).parallel().forEach(i -> {
			
			result.data[i] = (float) function.apply((double) input.data[i]).doubleValue();
			
		});
		
		return result;
		
	}
	
	public static Matrix dot(Matrix a, Matrix b, Matrix out) {
		
		assert a.columns == b.rows;
		
		int mdim, ndim, pdim;
		
		mdim = a.getRows();
		ndim = b.getColumns();
		pdim = a.getColumns();
		
		clSetKernelArg(dotKernel, 0, Sizeof.cl_int, Pointer.to(new int[] { mdim }));
		clSetKernelArg(dotKernel, 1, Sizeof.cl_int, Pointer.to(new int[] { ndim }));
		clSetKernelArg(dotKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { pdim }));
		clSetKernelArg(dotKernel, 3, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(dotKernel, 4, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(dotKernel, 5, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = ndim;
		global[1] = mdim;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, dotKernel, 2, null, global, null, 0, null, null);
		
		return out;
		
	}
	
	@Deprecated
	public static Matrix multiply(float a, Matrix b, Matrix out) {
		
		assert b.size == out.size;
		
		clSetKernelArg(dotKernel, 0, Sizeof.cl_float, Pointer.to(new float[] { a }));
		clSetKernelArg(dotKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(dotKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = b.size;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, fltMulKernel, 1, null, global, null, 0, null, null);
		
		
		return out;
		
	}
	
	public static Matrix multiply(Matrix a, Matrix b, Matrix out) {
		
		assert a.columns == b.columns && a.rows == b.rows && out.columns == a.columns && out.rows == a.rows;
		
		clSetKernelArg(dotKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(dotKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(dotKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, mulKernel, 1, null, global, null, 0, null, null);
		
		
		return out;
		
	}
	
	@Deprecated
	public static Matrix divide(Matrix a, Matrix b, Matrix out) {
		
		assert a.columns == b.columns && a.rows == b.rows;
		
		Matrix result = new Matrix(a.rows, b.columns);
		
		IntStream.range(0, result.rows * result.columns).parallel().forEach(i -> {
			result.data[i] = a.data[i] / b.data[i];
		});
		
		return result;
		
	}
	
	@Deprecated
	public static Matrix divide(float a, Matrix b, Matrix out) {
		
		Matrix result = new Matrix(b.rows, b.columns);
		
		IntStream.range(0, result.rows * result.columns).parallel().forEach(i -> {
			result.data[i] = a / b.data[i];
		});
		
		return result;
		
	}
	
	@Deprecated
	public static Matrix divide(Matrix a, float b, Matrix out) {
		
		Matrix result = new Matrix(a.rows, a.columns);
		
		IntStream.range(0, result.rows * result.columns).parallel().forEach(i -> {
			result.data[i] = a.data[i] / b;
		});
		
		return result;
		
	}
	
	public static Matrix transpose(Matrix a, Matrix out) {
		
		a.isTransposed = !a.isTransposed;
		
		float[] oldData = a.getData();
		float[] newData = new float[oldData.length];
		
		for (int row = 0; row < a.rows; row++) {
			for (int column = 0; column < a.columns; column++) {
				newData[column * a.rows + row] = oldData[row * a.columns + column];
			}
		}
		
		return new Matrix(a.columns, a.rows, newData);
	}
	
	public static Matrix add(Matrix a, Matrix b, Matrix out) {
		
		assert a.columns == b.columns && a.rows == b.rows && out.columns == a.columns && out.rows == a.rows;
		
		clSetKernelArg(addKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(addKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(addKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, addKernel, 1, null, global, null, 0, null, null);
		
		
		return out;
		
	}
	
	public static Matrix sub(Matrix a, Matrix b, Matrix out) {
		
		assert a.columns == b.columns && a.rows == b.rows && out.columns == a.columns && out.rows == a.rows;
		
		clSetKernelArg(subKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(subKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(subKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, subKernel, 1, null, global, null, 0, null, null);
		
		
		return out;
		
	}
	
	
	public static double sum(Matrix a) {
		
		float[] data = a.getData();
		
		return (double) IntStream.range(0, a.rows * a.columns).parallel().mapToDouble(i -> {
			return data[i];
		}).sum();
		
	}
	
	public static Matrix sigmoid(Matrix a, Matrix out) {
		
		assert out.size == a.size;
		
		clSetKernelArg(sigKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(sigKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, sigKernel, 1, null, global, null, 0, null, null);
		
		return out;
		
	}
	
	public static Matrix sigmoidPrime(Matrix a, Matrix out) {
		
		assert a.size == out.size;
		
		clSetKernelArg(sigPrimeKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(sigPrimeKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, sigPrimeKernel, 1, null, global, null, 0, null, null);
		
		return out;
	}
	
	@Override
	public String toString() {
		
		return rows + "x" + columns + " " + super.toString();
		
	}
	
}
