package core;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_ALL;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_WRITE;
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
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
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
	private static cl_kernel matFltDivKernel;
	private static cl_kernel fltMatDivKernel;
	private static cl_kernel matMatDivKernel;
	private static cl_kernel fltMulKernel;
	private static cl_kernel matPowKernel;
	private static cl_kernel dotATKernel;
	private static cl_kernel dotBTKernel;
	
	private static long[] global = new long[2];
	private static long[] local = new long[2];
	
	private int rows;
	private int columns;
	
	private int size;
	
	private transient cl_mem mData;
	
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
			
			mulKernel = loadKernel("matmul");
			dotKernel = loadKernel("matdot");
			sigKernel = loadKernel("matsig");
			sigPrimeKernel = loadKernel("matsigprime");
			addKernel = loadKernel("matadd");
			fltMulKernel = loadKernel("fltmul");
			matPowKernel = loadKernel("matpow");
			matFltDivKernel = loadKernel("matfltdiv");
			fltMatDivKernel = loadKernel("fltmatdiv");
			matMatDivKernel = loadKernel("matmatdiv");
			subKernel = loadKernel("matsub");
			dotATKernel = loadKernel("matdotat");
			dotBTKernel = loadKernel("matdotbt");
			
		} catch (IOException e) {
			
			e.printStackTrace();
			System.exit(1);
			
		}
		
	}
	
	private static cl_kernel loadKernel(String kernelName) throws IOException {
		
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File("kernel/" + kernelName + ".cl"))));
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
	
	public Matrix(int rows, int columns, float[] data) {
		
		assert data.length == rows * columns;
		
		this.rows = rows;
		this.columns = columns;
		this.size = rows * columns;
		
		Pointer pointer = Pointer.to(data);
		mData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * size, pointer, null);
		clEnqueueWriteBuffer(commandQueue, mData, CL_TRUE, 0, Sizeof.cl_float * size, pointer, 0, null, null);
		
	}
	
	public Matrix(int rows, int columns) {
		
		this(rows, columns, new float[rows * columns]);
		
	}
	
	public Matrix(int rows, int columns, Supplier<Double> generator) {
		
		this(rows, columns, arrayFromSupplier(rows * columns, generator));
		
	}
	
	public Matrix(Matrix a) {
		
		this.rows = a.rows;
		this.columns = a.columns;
		this.size = rows * columns;
		
		Pointer pointer = Pointer.to(a.mData);
		mData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * size, pointer, null);
		clEnqueueWriteBuffer(commandQueue, mData, CL_TRUE, 0, Sizeof.cl_float * size, pointer, 0, null, null);
		
	}
	
	public void release() {
		
		if (mData != null) {
			
			clReleaseMemObject(mData);
			mData = null;
			
		}
		
	}
	
	@Override
	protected void finalize() throws Throwable { // called by the garbage collector
		
		super.finalize();
		release();
		
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
	
	public float[] getData() {
		
		float[] data = new float[size];
		
		clEnqueueReadBuffer(commandQueue, mData, CL_TRUE, 0, Sizeof.cl_float * size, Pointer.to(data), 0, null, null);
		
		return data;
		
	}
	
	public void setData(float[] data) {
		
		clEnqueueWriteBuffer(commandQueue, mData, CL_TRUE, 0, Sizeof.cl_float * size, Pointer.to(data), 0, null, null);
		
	}
	
	@Override
	public String toString() {
		
		StringBuilder string = new StringBuilder();
		float[] data = getData();
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				string.append(data[row * columns + column] + "\t");
			}
			string.append("\n");
		}
		
		return string.toString();
		
	}
	
	private void writeObject(ObjectOutputStream out) throws IOException {
		
		out.writeInt(rows);
		out.writeInt(columns);
		float[] data = getData();
		out.writeObject(data);
		
	}
	
	private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
		
		rows = in.readInt();
		columns = in.readInt();
		size = rows * columns;
		float[] data = (float[]) in.readObject();
		mData = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * size, Pointer.to(data), null);
		setData(data);
		
	}
	
	public static Matrix dot(Matrix a, Matrix b, Matrix out, boolean aT, boolean bT) {
		
		assert !(aT && bT);
		
		int mdim, ndim, pdim;
		
		if (!aT && !bT) {
			
			assert a.columns == b.rows && out.rows == a.rows && out.columns == b.columns;
			
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
			
		} else if (aT) {
			
			assert a.rows == b.rows && out.rows == a.columns && out.columns == b.columns;
			
			mdim = a.getColumns();
			ndim = b.getColumns();
			pdim = a.getRows();
			
			clSetKernelArg(dotATKernel, 0, Sizeof.cl_int, Pointer.to(new int[] { mdim }));
			clSetKernelArg(dotATKernel, 1, Sizeof.cl_int, Pointer.to(new int[] { ndim }));
			clSetKernelArg(dotATKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { pdim }));
			clSetKernelArg(dotATKernel, 3, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
			clSetKernelArg(dotATKernel, 4, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
			clSetKernelArg(dotATKernel, 5, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
			
			global[0] = mdim;
			global[1] = ndim;
			local[0] = 1;
			
			clEnqueueNDRangeKernel(commandQueue, dotATKernel, 2, null, global, null, 0, null, null);
			
		} else if (bT) {
		
			assert a.columns == b.columns && out.rows == a.rows && out.columns == b.rows;
			
			mdim = a.getRows();
			ndim = b.getRows();
			pdim = a.getColumns();
			
			clSetKernelArg(dotBTKernel, 0, Sizeof.cl_int, Pointer.to(new int[] { mdim }));
			clSetKernelArg(dotBTKernel, 1, Sizeof.cl_int, Pointer.to(new int[] { ndim }));
			clSetKernelArg(dotBTKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { pdim }));
			clSetKernelArg(dotBTKernel, 3, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
			clSetKernelArg(dotBTKernel, 4, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
			clSetKernelArg(dotBTKernel, 5, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
			
			global[0] = mdim;
			global[1] = ndim;
			local[0] = 1;
			
			clEnqueueNDRangeKernel(commandQueue, dotBTKernel, 2, null, global, null, 0, null, null);
		}
		return out;
		
	}
	
	public static Matrix multiply(float a, Matrix b, Matrix out) {
		
		assert b.size == out.size;
		
		clSetKernelArg(fltMulKernel, 0, Sizeof.cl_float, Pointer.to(new float[] { a }));
		clSetKernelArg(fltMulKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(fltMulKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = b.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, fltMulKernel, 1, null, global, null, 0, null, null);
		
		return out;
		
	}
	
	public static Matrix multiply(Matrix a, Matrix b, Matrix out) {
		
		assert a.columns == b.columns && a.rows == b.rows && out.columns == a.columns && out.rows == a.rows;
		
		clSetKernelArg(mulKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(mulKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(mulKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, mulKernel, 1, null, global, null, 0, null, null);
		
		return out;
		
	}
	
	public static void pow(Matrix a, float b, Matrix out) {
		
		assert out.columns == a.columns && out.rows == a.rows;
		
		clSetKernelArg(matPowKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(matPowKernel, 1, Sizeof.cl_float, Pointer.to(new float[] { b }));
		clSetKernelArg(matPowKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, matPowKernel, 1, null, global, null, 0, null, null);
		
	}
	
	public static Matrix divide(Matrix a, Matrix b, Matrix out) {
		
		assert a.columns == b.columns && a.rows == b.rows && out.columns == a.columns && out.rows == a.rows;
		
		clSetKernelArg(matMatDivKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(matMatDivKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(matMatDivKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = b.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, matMatDivKernel, 1, null, global, null, 0, null, null);
		
		return out;
		
	}
	
	public static Matrix divide(float a, Matrix b, Matrix out) {
		
		assert out.columns == b.columns && out.rows == b.rows;
		
		clSetKernelArg(fltMatDivKernel, 0, Sizeof.cl_float, Pointer.to(new float[] { a }));
		clSetKernelArg(fltMatDivKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { b.mData }));
		clSetKernelArg(fltMatDivKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = b.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, fltMatDivKernel, 1, null, global, null, 0, null, null);
		
		return out;
		
	}
	
	public static Matrix divide(Matrix a, float b, Matrix out) {
		
		assert out.columns == a.columns && out.rows == a.rows;
		
		clSetKernelArg(matFltDivKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(matFltDivKernel, 1, Sizeof.cl_float, Pointer.to(new float[] { b }));
		clSetKernelArg(matFltDivKernel, 2, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, matFltDivKernel, 1, null, global, null, 0, null, null);
		
		return out;
		
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
	
	public static float sum(Matrix a) {
		
		float[] data = a.getData();
		
		return (float) IntStream.range(0, a.rows * a.columns).parallel().mapToDouble(i -> {
			return data[i];
		}).sum();
		
	}
	
	public static Matrix sigmoid(Matrix a, Matrix out) {
		
		assert a.rows == a.rows && a.columns == out.columns;
		
		clSetKernelArg(sigKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(sigKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, sigKernel, 1, null, global, null, 0, null, null);
		
		return out;
		
	}
	
	public static Matrix sigmoidPrime(Matrix a, Matrix out) {
		
		assert a.rows == a.rows && a.columns == out.columns;
		
		clSetKernelArg(sigPrimeKernel, 0, Sizeof.cl_mem, Pointer.to(new cl_mem[] { a.mData }));
		clSetKernelArg(sigPrimeKernel, 1, Sizeof.cl_mem, Pointer.to(new cl_mem[] { out.mData }));
		
		global[0] = a.size;
		local[0] = 1;
		
		clEnqueueNDRangeKernel(commandQueue, sigPrimeKernel, 1, null, global, null, 0, null, null);
		
		return out;
	}
	
	private static float[] arrayFromSupplier(int size, Supplier<Double> supplier) {
		
		float[] data = new float[size];
		for (int i = 0; i < size; i++) {
			data[i] = supplier.get().floatValue();
		}
		return data;
		
	}
	
}
