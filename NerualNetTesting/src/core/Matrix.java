package core;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class Matrix implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private static final NumberFormat FORMATTER = new DecimalFormat("#0.00");
	
	private int rows;
	private int columns;
	
	private double[][] data;
	
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
	
	public static Matrix dot(Matrix a, Matrix b) {
		
		assert a.columns == b.rows;
		
		Matrix result = new Matrix(a.rows, b.columns);
		
		IntStream.range(0, result.rows).parallel().forEach(row -> {
			for (int column = 0; column < result.columns; column++) {
				for (int i = 0; i < a.columns; i++) {
					result.data[row][column] += a.data[row][i] * b.data[i][column];
				}
			}
		});
		
		return result;
		
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
