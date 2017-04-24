
package core;

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
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class NeuralNetwork implements Serializable {
	
	private static final long serialVersionUID = 1L;
	
	private int inputLayerSize;
	private int outputLayerSize;
	// private int hiddenLayerSize;
	// private int numberOfHiddenLayers;
	
	private float overfittingPenalty = 0;
	
	private ArrayList<Matrix> w = new ArrayList<>();
	private transient ArrayList<Matrix> djdw = new ArrayList<>();
	private transient ArrayList<Matrix> z = new ArrayList<>();
	private transient ArrayList<Matrix> x = new ArrayList<>();
	
	public NeuralNetwork(NeuralNetwork network) {
		
		this.inputLayerSize = network.inputLayerSize;
		this.outputLayerSize = network.outputLayerSize;
		this.overfittingPenalty = network.overfittingPenalty;
		
		this.w = network.w.stream().map(Matrix::new).collect(Collectors.toCollection(ArrayList::new));
		
	}
	
	public NeuralNetwork(int inputLayerSize, int outputLayerSize, int hiddenLayerSize, int numberOfHiddenLayers, float overfittingPenalty, Random r) {
		
		assert numberOfHiddenLayers >= 1;
		
		this.inputLayerSize = inputLayerSize;
		this.outputLayerSize = outputLayerSize;
		// this.hiddenLayerSize = hiddenLayerSize;
		// this.numberOfHiddenLayers = numberOfHiddenLayers;
		
		this.overfittingPenalty = overfittingPenalty;
		
		w.add(new Matrix(hiddenLayerSize, inputLayerSize, r::nextGaussian));
		for (int i = 0; i < numberOfHiddenLayers - 1; i++) {
			w.add(new Matrix(hiddenLayerSize, hiddenLayerSize, r::nextGaussian));
		}
		w.add(new Matrix(outputLayerSize, hiddenLayerSize, r::nextGaussian));
		
		djdw = new ArrayList<>(w.size());
		x = new ArrayList<>();
		z = new ArrayList<>();
		
	}
	
	public Matrix forward(Matrix x0) {
		
		assert x0.getRows() == inputLayerSize;
		
		x.clear();
		z.clear();
		x.add(x0);
		
		for (int i = 0; i < w.size(); i++) {
			
			Matrix z_ = dot(w.get(i), x.get(i));
			x.add(Matrix.sigmoid(z_));
			z.add(z_);
			
		}
		
		return x.get(x.size() - 1);
		
	}
	
	public double getCost(Matrix x0, Matrix y) {
		return getCost(x0, y, forward(x0));
	}
	
	public double getCost(Matrix x, Matrix y, Matrix yHat) {
		
		return 0.5 * sum(map(sub(y, yHat), z -> z * z)) / x.getColumns() + overfittingPenalty * w.stream().mapToDouble(w -> sum(map(w, z -> z * z))).sum() / 2;
		
	}
	
	public ArrayList<Matrix> getCostPrime(Matrix x0, Matrix y) {
		return getCostPrime(x0, y, forward(x0));
	}
	
	public ArrayList<Matrix> getCostPrime(Matrix x0, Matrix y, Matrix yHat) {
		
		assert x0.getRows() == inputLayerSize && y.getRows() == outputLayerSize;
		
		ArrayList<Matrix> delta = new ArrayList<>();
		djdw.clear();
		for (int i = 0; i < w.size(); i++) {
			delta.add(null);
			djdw.add(null);
		}
		
		for (int i = w.size() - 1; i >= 0; i--) {
			
			if (i == w.size() - 1) {
				
				delta.set(i, multiply(sub(yHat, y), map(z.get(i), NeuralNetwork::activationPrime)));
				
			} else {
				
				delta.set(i, multiply(dot(transpose(w.get(i + 1)), delta.get(i + 1)), map(z.get(i), NeuralNetwork::activationPrime)));
				
			}
			djdw.set(i, dot(delta.get(i), transpose(x.get(i))));
			
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
			network.x = new ArrayList<>();
			network.z = new ArrayList<>();
			network.djdw = new ArrayList<>();
			
			ois.close();
			return network;
			
		} catch (ClassNotFoundException | IOException e) {
			
			e.printStackTrace();
			return null;
			
		}
		
	}
	
	public int getInputLayerSize() {
		
		return inputLayerSize;
		
	}
	
	public int getOutputLayerSize() {
		
		return outputLayerSize;
		
	}
	
}
