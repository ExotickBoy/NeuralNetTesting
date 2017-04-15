package core;

import static core.Matrix.add;
import static core.Matrix.*;
import static core.Matrix.map;
import static core.Matrix.multiply;
import static core.Matrix.sub;
import static core.Matrix.sum;
import static core.Matrix.transpose;
import static java.lang.Math.E;
import static java.lang.Math.pow;

import java.util.Random;

public class NN {
	
	private static double lambda = 0.0001;
	
	private int inputLayerSize;
	private int outputLayerSize;
	private int hiddenLayerSize;
	
	private Matrix w1;
	private Matrix w2;
	private Matrix w3;
	private Matrix w4;
	
	private Matrix a2;
	private Matrix a3;
	private Matrix a4;
	
	private Matrix z2;
	private Matrix z3;
	private Matrix z4;
	private Matrix z5;
	
	private Matrix dJdW1;
	private Matrix dJdW2;
	private Matrix dJdW3;
	private Matrix dJdW4;
	
	public NN(int inputLayerSize, int outputLayerSize, int hiddenLayerSize, Random r) {
		
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		
		w1 = new Matrix(inputLayerSize, hiddenLayerSize, () -> r.nextDouble() * .01);
		w2 = new Matrix(hiddenLayerSize, hiddenLayerSize, () -> r.nextDouble() * .01);
		w3 = new Matrix(hiddenLayerSize, hiddenLayerSize, () -> r.nextDouble() * .01);
		w4 = new Matrix(hiddenLayerSize, outputLayerSize, () -> r.nextDouble() * .01);
		
	}
	
	public Matrix forward(Matrix x) {
		
		assert x.getColumns() == inputLayerSize;
		
		z2 = dot(x, w1);
		a2 = map(z2, NN::activation);
		
		z3 = dot(a2, w2);
		a3 = map(z3, NN::activation);
		
		z4 = dot(a3, w3);
		a4 = map(z4, NN::activation);
		
		z5 = dot(a4, w4);
		Matrix yHat = map(z5, NN::activation);
		
		return yHat;
		
	}
	
	public double getCost(Matrix x, Matrix y, Matrix yHat) {
		
		return 0.5 * sum(map(sub(y, yHat), z -> z * z)) / x.getRows()
				+ (lambda / 2) * (sum(map(w1, z -> z * z)) + sum(map(w2, z -> z * z)) + sum(map(w3, z -> z * z)) + sum(map(w4, z -> z * z)));
		
	}
	
	public void findCostPrime(Matrix x, Matrix y) {
		findCostPrime(x, y, forward(x));
	}
	
	public void findCostPrime(Matrix x, Matrix y, Matrix yHat) {
		
		assert x.getColumns() == inputLayerSize && y.getColumns() == outputLayerSize;
		
		Matrix delta5 = multiply(sub(yHat, y), map(z5, NN::activationPrime));
		dJdW4 = add(dot(transpose(a4), delta5), multiply(lambda, w4));
		
		Matrix delta4 = multiply(dot(delta5, transpose(w4)), map(z4, NN::activationPrime));
		dJdW3 = add(dot(transpose(a3), delta4), multiply(lambda, w3));
		
		Matrix delta3 = multiply(dot(delta4, transpose(w3)), map(z2, NN::activationPrime));
		dJdW2 = add(dot(transpose(a2), delta3), multiply(lambda, w2));
		
		Matrix delta2 = multiply(dot(delta3, transpose(w2)), map(z2, NN::activationPrime));
		dJdW1 = add(dot(transpose(x), delta2), multiply(lambda, w1));
		
	}
	
	public void descend(double learningRate) {
		
		w1 = sub(w1, multiply(learningRate, dJdW1));
		w2 = sub(w2, multiply(learningRate, dJdW2));
		w3 = sub(w3, multiply(learningRate, dJdW3));
		w4 = sub(w4, multiply(learningRate, dJdW4));
		
	}
	
	private static double activation(double x) {
		return 1 / (1 + pow(E, -x));
		// return tanh(x);
	}
	
	private static double activationPrime(double x) {
		return activation(x) * (1 - activation(x));
		// return pow(cosh(x), -2);
	}
	
}
