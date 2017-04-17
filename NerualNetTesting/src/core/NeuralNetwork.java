
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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class NeuralNetwork implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private int inputLayerSize;
	private int outputLayerSize;
	// private int hiddenLayerSize;
	// private int numberOfHiddenLayers;
	
	private double overfittingPenalty = 0;
	
	private ArrayList<Matrix> w = new ArrayList<>(); // -1
	private transient ArrayList<Matrix> djdw = new ArrayList<>(); // -1
	private transient ArrayList<Matrix> a = new ArrayList<>(); // -2
	private transient ArrayList<Matrix> z = new ArrayList<>(); // -2
	
	public NeuralNetwork(int inputLayerSize, int outputLayerSize, int hiddenLayerSize, int numberOfHiddenLayers, double overfittingPenalty, Random r) {
		
		assert numberOfHiddenLayers >= 1;
		
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
		// this.hiddenLayerSize = hiddenLayerSize;
		// this.numberOfHiddenLayers = numberOfHiddenLayers;
		
		this.overfittingPenalty = overfittingPenalty;
		
		w.add(new Matrix(inputLayerSize, hiddenLayerSize, r::nextGaussian));
		for (int i = 0; i < numberOfHiddenLayers - 1; i++) {
			w.add(new Matrix(hiddenLayerSize, hiddenLayerSize, r::nextGaussian));
		}
		w.add(new Matrix(hiddenLayerSize, outputLayerSize, r::nextGaussian));
		
		djdw = new ArrayList<>(w.size());
		
	}
	
	public Matrix forward(Matrix x) {
		
		assert x.getColumns() == inputLayerSize;
		
		Matrix yHat = null;
		for (int i = 0; i < w.size(); i++) {
			
			if (i == 0) {
				
				z.add(dot(x, w.get(i)));
				a.add(map(z.get(z.size() - 1), NeuralNetwork::activation));
				
			} else if (i != w.size() - 1) {
				
				z.add(dot(a.get(a.size() - 1), w.get(i)));
				a.add(map(z.get(z.size() - 1), NeuralNetwork::activation));
				
			} else {
				
				z.add(dot(a.get(a.size() - 1), w.get(i)));
				yHat = map(z.get(z.size() - 1), NeuralNetwork::activation);
				
			}
			
		}
		
		return yHat;
		
	}
	
	public double getCost(Matrix x, Matrix y) {
		return getCost(x, y, forward(x));
	}
	
	public double getCost(Matrix x, Matrix y, Matrix yHat) {
		
		return 0.5 * sum(map(sub(y, yHat), z -> z * z)) / x.getRows() + overfittingPenalty * w.stream().mapToDouble(w -> sum(map(w, z -> z * z))).sum() / 2;
		
	}
	
	public ArrayList<Matrix> getCostPrime(Matrix x, Matrix y) {
		return getCostPrime(x, y, forward(x));
	}
	
	public ArrayList<Matrix> getCostPrime(Matrix x, Matrix y, Matrix yHat) {
		
		assert x.getColumns() == inputLayerSize && y.getColumns() == outputLayerSize;
		
		ArrayList<Matrix> delta = new ArrayList<>();
		djdw = new ArrayList<>();
		for (int i = 0; i < w.size(); i++) {
			delta.add(null);
			djdw.add(null);
		}
		
		for (int i = w.size() - 1; i >= 0; i--) {
			
			if (i == w.size() - 1) {
				
				delta.set(i, multiply(sub(yHat, y), map(z.get(i), NeuralNetwork::activationPrime)));
				djdw.set(i, add(dot(transpose(a.get(i - 1)), delta.get(i)), multiply(overfittingPenalty, w.get(i))));
				
			} else if (i != 0) {
				
				delta.set(i, multiply(dot(delta.get(i + 1), transpose(w.get(i + 1))), map(z.get(i), NeuralNetwork::activationPrime)));
				djdw.set(i, add(dot(transpose(a.get(i - 1)), delta.get(i)), multiply(overfittingPenalty, w.get(i))));
				
			} else {
				
				delta.set(i, multiply(dot(delta.get(i + 1), transpose(w.get(i + 1))), map(z.get(i), NeuralNetwork::activationPrime)));
				djdw.set(i, add(dot(transpose(x), delta.get(i)), multiply(overfittingPenalty, w.get(i))));
				
			}
			
		}
		
		return djdw;
		
	}
	
	private static double activation(double x) {
		return 1 / (1 + pow(E, -x));
		// return tanh(x);
	}
	
	private static double activationPrime(double x) {
		return activation(x) * (1 - activation(x));
		// return pow(cosh(x), -2);
	}
	
	public ArrayList<Matrix> getW() {
		
		return w;
		
	}
	
	public void setW(ArrayList<Matrix> w) {
		
		this.w = w;
		
	}
	
	public void save(File file) {
		
		try {
			
			file.getParentFile().mkdirs();
			ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(file)));
			oos.writeObject(this);
			oos.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static NeuralNetwork load(File file) {
		
		try {
			
			ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(new FileInputStream(file)));
			NeuralNetwork network = (NeuralNetwork) ois.readObject();
			ois.close();
			return network;
			
		} catch (ClassNotFoundException | IOException e) {
			
			e.printStackTrace();
			return null;
			
		}
		
	}
	
}
