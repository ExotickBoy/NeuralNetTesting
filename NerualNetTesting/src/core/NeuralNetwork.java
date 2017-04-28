
package core;

import static core.Matrix.dot;
import static core.Matrix.map;
import static core.Matrix.multiply;
import static core.Matrix.sigmoidPrime;
import static core.Matrix.sub;
import static core.Matrix.sum;
import static core.Matrix.transpose;
import static java.lang.Math.E;
import static java.lang.Math.pow;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class NeuralNetwork implements Serializable {
	
	private static final long serialVersionUID = 2L;
	
	private int inputLayerSize;
	private int outputLayerSize;
	private int hiddenLayerSize;
	private int numberOfHiddenLayers;
	
	private transient Matrix[] w;
	private transient Matrix[] djdw;
	private transient Matrix[] x;
	private transient Matrix[] delta;
	
	public NeuralNetwork(NeuralNetwork network) {
		
		this.inputLayerSize = network.inputLayerSize;
		this.outputLayerSize = network.outputLayerSize;
		this.hiddenLayerSize = network.hiddenLayerSize;
		this.numberOfHiddenLayers = network.numberOfHiddenLayers;
		
		w = new Matrix[numberOfHiddenLayers + 1];
		x = new Matrix[numberOfHiddenLayers + 2];
		djdw = new Matrix[w.length];
		delta = new Matrix[w.length];
		
		for (int i = 0; i < w.length; i++) {
			w[i] = new Matrix(network.w[i]);
		}
		
	}
	
	public NeuralNetwork(int inputLayerSize, int outputLayerSize, int hiddenLayerSize, int numberOfHiddenLayers, Random r) {
		
		assert numberOfHiddenLayers >= 1;
		
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.numberOfHiddenLayers = numberOfHiddenLayers;
		
		w = new Matrix[numberOfHiddenLayers + 1];
		x = new Matrix[numberOfHiddenLayers + 2];
		djdw = new Matrix[w.length];
		delta = new Matrix[w.length];
		
		w[0] = new Matrix(hiddenLayerSize, inputLayerSize, r::nextGaussian);
		for (int i = 1; i < numberOfHiddenLayers; i++) {
			w[i] = new Matrix(hiddenLayerSize, hiddenLayerSize, r::nextGaussian);
		}
		w[numberOfHiddenLayers] = new Matrix(outputLayerSize, hiddenLayerSize, r::nextGaussian);
		
	}
	
	public Matrix forward(Matrix x0) {
		
		assert x0.getRows() == inputLayerSize;
		
		x[0] = x0;
		for (int i = 0; i < w.length; i++) {
			
			x[i + 1] = Matrix.sigmoid(dot(w[i], x[i]));
			
		}
		
		return x[numberOfHiddenLayers + 1];
		
	}
	
	public double getCost(Matrix x0, Matrix y) {
		
		return getCost(x0, y, forward(x0));
		
	}
	
	public double getCost(Matrix x, Matrix y, Matrix yHat) {
		
		return 0.5 * sum(map(sub(y, yHat), z -> z * z)) / x.getColumns();
		
	}
	
	public Matrix[] getCostPrime(Matrix x0, Matrix y) {
		
		return getCostPrime(x0, y, forward(x0));
		
	}
	
	public Matrix[] getCostPrime(Matrix x0, Matrix y, Matrix yHat) {
		
		assert x0.getRows() == inputLayerSize && y.getRows() == outputLayerSize;
		
		for (int i = w.length - 1; i >= 0; i--) {
			
			if (i == w.length - 1) {
				
				delta[i] = multiply(sub(yHat, y), sigmoidPrime(x[i + 1]));
				
			} else {
				
				delta[i] = multiply(dot(transpose(w[i + 1]), delta[i + 1]), sigmoidPrime(x[i + 1]));
				
			}
			
			djdw[i] = dot(delta[i], transpose(x[i]));
			
		}
		
		return djdw;
		
	}
	
	public static double activation(double x) {
		
		return 1 / (1 + pow(E, -x));
		
	}
	
	public static double activationPrime(double x) {
		
		double a = activation(x);
		return a * (1 - a);
		
	}
	
	public Matrix[] getW() {
		
		return w;
		
	}
	
	public void setW(Matrix[] w) {
		
		this.w = w;
		
	}
	
	public void save(File file) throws IOException {
		
		file.getParentFile().mkdirs();
		ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(file)));
		oos.writeObject(this);
		oos.close();
		
	}
	
	public static NeuralNetwork load(File file) throws IOException, ClassNotFoundException {
		
		ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(new FileInputStream(file)));
		NeuralNetwork network = (NeuralNetwork) ois.readObject();
		ois.close();
		
		return network;
		
	}
	
	public int getInputLayerSize() {
		
		return inputLayerSize;
		
	}
	
	public int getOutputLayerSize() {
		
		return outputLayerSize;
		
	}
	
}
