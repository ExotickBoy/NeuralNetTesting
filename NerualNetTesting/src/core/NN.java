<<<<<<< HEAD
package core;

import static core.Matrix.add;
import static core.Matrix.dot;
import static core.Matrix.map;
import static core.Matrix.multiply;
import static core.Matrix.sub;
import static core.Matrix.sum;
import static core.Matrix.transpose;
import static java.lang.Math.E;
import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NN {
	
	private static double lambda = 0.0001; // overfitting penalty
	
	private int inputLayerSize;
	private int outputLayerSize;
	private int hiddenLayerSize;
	private int numberOfHiddenLayers;
	
	private ArrayList<Matrix> w = new ArrayList<>(); // -1
	private ArrayList<Matrix> djdw = new ArrayList<>(); // -1
	private ArrayList<Matrix> a = new ArrayList<>(); // -2
	private ArrayList<Matrix> z = new ArrayList<>(); // -2
	
	public NN(int inputLayerSize, int outputLayerSize, int hiddenLayerSize, int numberOfHiddenLayers, Random r) {
		
		assert numberOfHiddenLayers >= 1;
		
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.numberOfHiddenLayers = numberOfHiddenLayers;
		
		w.add(new Matrix(inputLayerSize, hiddenLayerSize, r::nextDouble));
		for (int i = 0; i < numberOfHiddenLayers - 1; i++) {
			w.add(new Matrix(hiddenLayerSize, hiddenLayerSize, r::nextDouble));
		}
		w.add(new Matrix(hiddenLayerSize, outputLayerSize, r::nextDouble));
		
		djdw = new ArrayList<>(w.size());
		
	}
	
	public Matrix forward(Matrix x) {
		
		assert x.getColumns() == inputLayerSize;
		
		Matrix yHat = null;
		for (int i = 0; i < w.size(); i++) {
			
			if (i == 0) {
				
				z.add(dot(x, w.get(i)));
				a.add(map(z.get(z.size() - 1), NN::activation));
				
			} else if (i != w.size() - 1) {
				
				z.add(dot(a.get(a.size() - 1), w.get(i)));
				a.add(map(z.get(z.size() - 1), NN::activation));
				
			} else {
				
				z.add(dot(a.get(a.size() - 1), w.get(i)));
				yHat = map(z.get(z.size() - 1), NN::activation);
				
			}
			
		}
		
		return yHat;
		
	}
	
	public double getCost(Matrix x, Matrix y, Matrix yHat) {
		
		return 0.5 * sum(map(sub(y, yHat), z -> z * z)) / x.getRows() + lambda * w.stream().mapToDouble(w -> sum(map(w, z -> z * z))).sum() / 2;
		
	}
	
	public void findCostPrime(Matrix x, Matrix y) {
		findCostPrime(x, y, forward(x));
	}
	
	public void findCostPrime(Matrix x, Matrix y, Matrix yHat) {
		
		assert x.getColumns() == inputLayerSize && y.getColumns() == outputLayerSize;
		
		ArrayList<Matrix> delta = new ArrayList<>();
		djdw = new ArrayList<>();
		for (int i = 0; i < w.size(); i++) {
			delta.add(null);
			djdw.add(null);
		}
		
		for (int i = w.size() - 1; i >= 0; i--) {
						
			if (i == w.size() - 1) {
				
				delta.set(i, multiply(sub(yHat, y), map(z.get(i), NN::activationPrime)));
				djdw.set(i, add(dot(transpose(a.get(i - 1)), delta.get(i)), multiply(lambda, w.get(i))));
				
			} else if (i != 0) {
				
				delta.set(i, multiply(dot(delta.get(i + 1), transpose(w.get(i + 1))), map(z.get(i), NN::activationPrime)));
				djdw.set(i, add(dot(transpose(a.get(i - 1)), delta.get(i)), multiply(lambda, w.get(i))));
				
			} else {
				
				delta.set(i, multiply(dot(delta.get(i + 1), transpose(w.get(i + 1))), map(z.get(i), NN::activationPrime)));
				djdw.set(i, add(dot(transpose(x), delta.get(i)), multiply(lambda, w.get(i))));
				
			}
			
		}
				
	}
	
	public void descend(double learningRate) {
		
		IntStream.range(0, w.size()).parallel().forEach(i -> {
			w.set(i, sub(w.get(i), multiply(learningRate, djdw.get(i))));
		});
		
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
=======
package core;

import static core.Matrix.add;
import static core.Matrix.dot;
import static core.Matrix.map;
import static core.Matrix.multiply;
import static core.Matrix.sub;
import static core.Matrix.sum;
import static core.Matrix.transpose;
import static java.lang.Math.E;
import static java.lang.Math.pow;

import java.util.ArrayList;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NN {
	
	private static double lambda = 0.0001; // overfitting penalty
	
	private int inputLayerSize;
	private int outputLayerSize;
	private int hiddenLayerSize;
	private int numberOfHiddenLayers;
	
	private ArrayList<Matrix> w = new ArrayList<>(); // -1
	private ArrayList<Matrix> djdw = new ArrayList<>(); // -1
	private ArrayList<Matrix> a = new ArrayList<>(); // -2
	private ArrayList<Matrix> z = new ArrayList<>(); // -2
	
	public NN(int inputLayerSize, int outputLayerSize, int hiddenLayerSize, int numberOfHiddenLayers, Random r) {
		
		assert numberOfHiddenLayers >= 1;
		
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.numberOfHiddenLayers = numberOfHiddenLayers;
		
		w.add(new Matrix(inputLayerSize, hiddenLayerSize, r::nextDouble));
		for (int i = 0; i < numberOfHiddenLayers - 1; i++) {
			w.add(new Matrix(hiddenLayerSize, hiddenLayerSize, r::nextDouble));
		}
		w.add(new Matrix(hiddenLayerSize, outputLayerSize, r::nextDouble));
		
		djdw = new ArrayList<>(w.size());
		
	}
	
	public Matrix forward(Matrix x) {
		
		assert x.getColumns() == inputLayerSize;
		
		Matrix yHat = null;
		for (int i = 0; i < w.size(); i++) {
			
			if (i == 0) {
				
				z.add(dot(x, w.get(i)));
				a.add(map(z.get(z.size() - 1), NN::activation));
				
			} else if (i != w.size() - 1) {
				
				z.add(dot(a.get(a.size() - 1), w.get(i)));
				a.add(map(z.get(z.size() - 1), NN::activation));
				
			} else {
				
				z.add(dot(a.get(a.size() - 1), w.get(i)));
				yHat = map(z.get(z.size() - 1), NN::activation);
				
			}
			
		}
		
		return yHat;
		
	}
	
	public double getCost(Matrix x, Matrix y, Matrix yHat) {
		
		return 0.5 * sum(map(sub(y, yHat), z -> z * z)) / x.getRows() + lambda * w.stream().mapToDouble(w -> sum(map(w, z -> z * z))).sum() / 2;
		
	}
	
	public void findCostPrime(Matrix x, Matrix y) {
		findCostPrime(x, y, forward(x));
	}
	
	public void findCostPrime(Matrix x, Matrix y, Matrix yHat) {
		
		assert x.getColumns() == inputLayerSize && y.getColumns() == outputLayerSize;
		
		ArrayList<Matrix> delta = new ArrayList<>();
		djdw = new ArrayList<>();
		for (int i = 0; i < w.size(); i++) {
			delta.add(null);
			djdw.add(null);
		}
		
		for (int i = w.size() - 1; i >= 0; i--) {
						
			if (i == w.size() - 1) {
				
				delta.set(i, multiply(sub(yHat, y), map(z.get(i), NN::activationPrime)));
				djdw.set(i, add(dot(transpose(a.get(i - 1)), delta.get(i)), multiply(lambda, w.get(i))));
				
			} else if (i != 0) {
				
				delta.set(i, multiply(dot(delta.get(i + 1), transpose(w.get(i + 1))), map(z.get(i), NN::activationPrime)));
				djdw.set(i, add(dot(transpose(a.get(i - 1)), delta.get(i)), multiply(lambda, w.get(i))));
				
			} else {
				
				delta.set(i, multiply(dot(delta.get(i + 1), transpose(w.get(i + 1))), map(z.get(i), NN::activationPrime)));
				djdw.set(i, add(dot(transpose(x), delta.get(i)), multiply(lambda, w.get(i))));
				
			}
			
		}
				
	}
	
	public void descend(double learningRate) {
		
		IntStream.range(0, w.size()).parallel().forEach(i -> {
			w.set(i, sub(w.get(i), multiply(learningRate, djdw.get(i))));
		});
		
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
>>>>>>> branch 'master' of https://github.com/ExotickBoy/NeuralNetTesting.git
