package core;

import static core.Matrix.*;

import java.util.ArrayList;
import java.util.Random;
import java.util.stream.Collectors;

import static java.lang.Math.*;

public class NumericalGradientChecking {
	
	private static final double EPSILON = 1e-4;
	
	public static void main(String[] args) {
		
		// Matrix x = new Matrix(2, 5, new float[] { .1f, .9f, .4f, .5f, .6f, .4f, .1f, .3f, .9f, .1f, });
		// Matrix y = new Matrix(1, 5, new float[] { .1f, .4f, .3f, .5f, .9f });
		
		Matrix x = new Matrix(2, 1, new float[] { 0f, 0f, });
		Matrix y = new Matrix(1, 1, new float[] { 1f });
		
		Random r = new Random();
		NeuralNetwork network = new NeuralNetwork(2, 1, 5, 1, 0, r);
		
		ArrayList<Matrix> ws = network.getW().stream().map(Matrix::new).collect(Collectors.toCollection(ArrayList::new));
		
		Matrix yHat = network.forward(x);
		double defaultCost = network.getCost(x, y, yHat);
		
		ArrayList<Matrix> djdw = network.getCostPrime(x, y, yHat);
		ArrayList<Matrix> djdw2 = djdw.stream().map(Matrix::new).collect(Collectors.toCollection(ArrayList::new));
		
		for (int matrix = 0; matrix < djdw.size(); matrix++) {
			for (int row = 0; row < djdw.get(matrix).getRows(); row++) {
				for (int column = 0; column < djdw.get(matrix).getColumns(); column++) {
					
					ArrayList<Matrix> wn = ws.stream().map(Matrix::new).collect(Collectors.toCollection(ArrayList::new));
					wn.get(matrix).set(row, column, ws.get(matrix).get(row, column) + EPSILON);
					network.setW(wn);
					double newCost = network.getCost(x, y);
					
					djdw2.get(matrix).set(row, column, (newCost - defaultCost) / EPSILON);
					
				}
			}
		}
		
		for (int matrix = 0; matrix < ws.size(); matrix++) {
			System.out.println(norm(sub(djdw.get(matrix), djdw2.get(matrix))) / norm(add(djdw.get(matrix), djdw2.get(matrix))));
		}
		
	}
	
	public static double norm(Matrix x) {
		return sum(map(x, n -> abs(n)));
	}
	
}
