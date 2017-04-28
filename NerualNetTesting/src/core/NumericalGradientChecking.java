package core;

import static core.Matrix.add;
import static core.Matrix.map;
import static core.Matrix.sub;
import static core.Matrix.sum;
import static java.lang.Math.abs;

import java.util.Random;

public class NumericalGradientChecking {
	
	private static final double EPSILON = 1e-4;
	
	public static void main(String[] args) {
		
		// Matrix x = new Matrix(2, 5, new float[] { .1f, .9f, .4f, .5f, .6f, .4f, .1f, .3f, .9f, .1f, });
		// Matrix y = new Matrix(1, 5, new float[] { .1f, .4f, .3f, .5f, .9f });
		
		Matrix x = new Matrix(2, 1, new float[] { 0f, 0f, });
		Matrix y = new Matrix(1, 1, new float[] { 1f });
		
		Random r = new Random();
		NeuralNetwork network = new NeuralNetwork(2, 1, 5, 1, r);
		
		Matrix[] djdw = network.getCostPrime(x, y);
		Matrix[] djdw2 = new Matrix[djdw.length];
		
		Matrix[] initialWeights = new Matrix[network.getW().length];
		Matrix[] perturbedWeights = new Matrix[network.getW().length];
		for (int i = 0; i < network.getW().length; i++) {
			initialWeights[i] = new Matrix(network.getW()[i]);
			perturbedWeights[i] = new Matrix(network.getW()[i]);
			djdw2[i] = new Matrix(djdw[i]);
		}
		
		
		network.setW(perturbedWeights);
		
		for (int matrix = 0; matrix < djdw.length; matrix++) {
			for (int row = 0; row < djdw[matrix].getRows(); row++) {
				for (int column = 0; column < djdw[matrix].getColumns(); column++) {
					
					perturbedWeights[matrix].set(row, column, initialWeights[matrix].get(row, column) + EPSILON);
					double loss2 = network.getCost(x, y);
					
					perturbedWeights[matrix].set(row, column, initialWeights[matrix].get(row, column) - EPSILON);
					double loss1 = network.getCost(x, y);
					
					djdw2[matrix].set(row, column, (loss2 - loss1) / (2 * EPSILON));
					
				}
			}
		}
		
		for (int matrix = 0; matrix < initialWeights.length; matrix++) {
			System.out.println(norm(sub(djdw[matrix], djdw2[matrix])) / norm(add(djdw[matrix], djdw2[matrix])));
		}
		
	}
	
	public static double norm(Matrix x) {
		return sum(map(x, n -> abs(n)));
	}
	
}
