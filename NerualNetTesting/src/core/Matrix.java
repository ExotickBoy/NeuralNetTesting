package core;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
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

public class Matrix implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private static final NumberFormat FORMATTER = new DecimalFormat("#0.00");
	
	private static cl_context context;
	private static cl_command_queue commandQueue;
	
	private static cl_program program;
	private static cl_kernel kernel;
	
	private static long[] global = new long[2];
	private static long[] local = new long[2];
	
	private int rows;
	private int columns;
	
	private double[][] data;
	
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
        context = clCreateContext(
            contextProperties, 1, new cl_device_id[]{device}, 
            null, null, null);

		
	
        // Create a command-queue for the selected device
        commandQueue = 
            clCreateCommandQueue(context, device, 0, null);
        
        String code = loadCLCode("kernel/mat_mul.cl");
		
		program = clCreateProgramWithSource(context, 1, new String[]{code}, null, null);
		 clBuildProgram(program, 0, null, null, null, null);
		
		kernel = clCreateKernel(program, "matmul", null);
	}
	
	public Matrix(int rows, int columns) {
		
		this(rows, columns, new double[rows][columns]);
		
	}
	
	public Matrix(int rows, int columns, double[][] data) {
		
		this.rows = rows;
		this.columns = columns;
		
		this.data = data;
		
	}
	
	public Matrix(int rows, int columns, Supplier<Double> generator) {
		
		this(rows, columns);
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				data[row][column] = generator.get();
			}
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
	
	public void set(int row, int column, double a) {
		
		data[row][column] = a;
		
	}
	
	public double get(int row, int column) {
		
		return data[row][column];
		
	}
	
	public static Matrix map(Matrix input, Function<Double, Double> function) {
		
		Matrix result = new Matrix(input.rows, input.columns);
		
		IntStream.range(0, input.rows).parallel().forEach(row -> {
			for (int column = 0; column < input.columns; column++) {
				result.data[row][column] = function.apply(input.data[row][column]);
			}
		});
		
		return result;
		
	}
	
	public static Matrix dot(Matrix a, Matrix b){
		
		int mdim, ndim, pdim;
		
		mdim = a.getRows();
		ndim = b.getColumns();
		pdim = a.getColumns();
		
		int szA, szB, szC;
		
		szA = mdim * pdim;
		szB = pdim * ndim;
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
	
	public static Matrix multiply(double a, Matrix b) {
		
		Matrix result = new Matrix(b.rows, b.columns);
		
		IntStream.range(0, result.rows).parallel().forEach(row -> {
			for (int column = 0; column < result.columns; column++) {
				result.data[row][column] = a * b.data[row][column];
			}
		});
		
		return result;
		
	}
	
	public static Matrix multiply(Matrix a, Matrix b) {
		
		assert a.columns == b.columns && a.rows == b.rows;
		
		Matrix result = new Matrix(a.rows, b.columns);
		
		IntStream.range(0, result.rows).parallel().forEach(row -> {
			for (int column = 0; column < result.columns; column++) {
				result.data[row][column] = a.data[row][column] * b.data[row][column];
			}
		});
		
		return result;
		
	}
	
	public static Matrix transpose(Matrix a) {
		Matrix result = new Matrix(a.columns, a.rows);
		IntStream.range(0, result.rows).parallel().forEach(row -> {
			for (int column = 0; column < result.columns; column++) {
				result.data[row][column] = a.data[column][row];
			}
		});
		return result;
	}
	
	public static Matrix add(Matrix a, Matrix b) {
		
		assert a.rows == b.rows && a.columns == b.columns;
		
		Matrix result = new Matrix(a.rows, a.columns);
		IntStream.range(0, result.rows).parallel().forEach(row -> {
			for (int column = 0; column < result.columns; column++) {
				result.data[row][column] = a.data[row][column] + b.data[row][column];
			}
		});
		
		return result;
		
	}
	
	public static Matrix sub(Matrix a, Matrix b) {
		
		assert a.rows == b.rows && a.columns == b.columns;
		
		Matrix result = new Matrix(a.rows, a.columns);
		IntStream.range(0, result.rows).parallel().forEach(row -> {
			for (int column = 0; column < a.columns; column++) {
				result.data[row][column] = a.data[row][column] - b.data[row][column];
			}
		});
		
		return result;
		
	}
	
	public static double sum(Matrix a) {
		
		return IntStream.range(0, a.rows).parallel().mapToDouble(row -> {
			return IntStream.range(0, a.columns).parallel().mapToDouble(column -> {
				return a.data[row][column];
			}).sum();
		}).sum();
		
	}
	
	@Override
	public String toString() {
		
		StringBuilder string = new StringBuilder();
		
		for (int row = 0; row < rows; row++) {
			for (int column = 0; column < columns; column++) {
				string.append(FORMATTER.format(data[row][column]) + ",\t");
			}
			string.append("\n");
		}
		
		return string.toString();
		
	}
	
}
