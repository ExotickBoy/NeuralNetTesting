
package core;

import static core.Matrix.dot;
import static core.Matrix.map;
import static core.Matrix.multiply;
import static core.Matrix.sigmoidPrime;
import static core.Matrix.sub;
import static core.Matrix.sum;
import static core.Matrix.transpose;

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

/**
 * 
 * This class is an implementation of a NeuralNetwork
 * 
 * @author Kacper, Jamie
 *
 */
public class NeuralNetwork implements Serializable {
	
	private static final long serialVersionUID = 3L;
	
	private int inputLayerSize;
	private int outputLayerSize;
	private int hiddenLayerSize;
	private int numberOfHiddenLayers;
	
	private transient Matrix[] w;
	private transient Matrix[] djdw;
	private transient Matrix[] x;
	private transient Matrix[] delta;
	
	/**
	 * 
	 * Creates a new NeuralNetwork which is a copy of the network passed to it
	 * 
	 * @param network
	 *            - network to be cloned
	 */
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
	
	/**
	 * Creates a new NeuralNetwork with random weights
	 * 
	 * @see trainers.TrainingScheme
	 * 
	 * @param inputLayerSize
	 *            - dimensions of input data
	 * @param outputLayerSize
	 *            - dimensions of output data
	 * @param hiddenLayerSize
	 *            - dimensions of hidden layers
	 * @param numberOfHiddenLayers
	 *            - amount of hidden layers
	 * @param r
	 *            - random object that will be used for the initial weights
	 */
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
	
	/**
	 * 
	 * Forwards the data x0 through the network and returns the resultant Matrix y
	 * 
	 * @param x0
	 *            - input data
	 * @return output data
	 */
	public Matrix forward(Matrix x0) {
		
		assert x0.getRows() == inputLayerSize;
		
		x[0] = x0;
		for (int i = 0; i < w.length; i++) {
			
			x[i + 1] = Matrix.sigmoid(dot(w[i], x[i]));
			
		}
		
		return x[numberOfHiddenLayers + 1];
		
	}
	
	/**
	 * 
	 * Finds the cost(error) of the network in it's current state. The cost is equal to the half the
	 * sum of the squares of the difference between the output the network produces and the expected
	 * output
	 * 
	 * 
	 * @param x0
	 *            - the input data
	 * @param y
	 *            - the expected output
	 * @return the cost
	 */
	public double getCost(Matrix x0, Matrix y) {
		
		return getCost(x0, y, forward(x0));
		
	}
	
	/**
	 * Finds the cost(error) of the network in it's current state. The cost is equal to the half the
	 * sum of the squares of the difference between the output the network produces and the expected
	 * output
	 * 
	 * @param x
	 *            - the input data
	 * @param y
	 *            - the expected output data
	 * @param yHat
	 *            - the actual output data
	 * @return the cost
	 */
	public double getCost(Matrix x, Matrix y, Matrix yHat) {
		
		return 0.5 * sum(map(sub(y, yHat), z -> z * z)) / x.getColumns();
		
	}
	
	/**
	 * 
	 * Finds the partial derivative of the cost and the weights of the network.
	 * 
	 * @param x0
	 *            - the input data
	 * @param y
	 *            - the expected output data
	 * @return an array containing the partial derivatives of the weights
	 */
	public Matrix[] getCostPrime(Matrix x0, Matrix y) {
		
		return getCostPrime(x0, y, forward(x0));
		
	}
	
	/**
	 * 
	 * Finds the partial derivative of the cost and the weights of the network.
	 * 
	 * @param x0
	 *            - the input data
	 * @param y
	 *            - the expected output
	 * @param yHat
	 *            - the actual output
	 * @return an array containing the partial derivatives of the weights
	 */
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
	
	/**
	 * 
	 * @return the dimensions of the input column vectors
	 */
	public int getInputLayerSize() {
		
		return inputLayerSize;
		
	}
	
	/**
	 * 
	 * @return the dimensions of the output vectors
	 */
	public int getOutputLayerSize() {
		
		return outputLayerSize;
		
	}
	
	/**
	 * 
	 * @return the list of matrices which are the weights of the neural network
	 */
	public Matrix[] getW() {
		
		return w;
		
	}
	
	/**
	 * 
	 * @param w
	 *            - the new weights for the network
	 */
	public void setW(Matrix[] w) {
		
		this.w = w;
		
	}
	
	/**
	 * 
	 * Writes the network to a file
	 * 
	 * @param file
	 *            - where the network should be written
	 * @throws IOException
	 */
	public void save(File file) throws IOException {
		
		file.getParentFile().mkdirs();
		ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(file)));
		oos.writeObject(this);
		oos.writeObject(w);
		oos.close();
		
	}
	
	/**
	 * 
	 * Reads a network from a file
	 * 
	 * @param file
	 *            - where to read from
	 * @return the network found in the file
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static NeuralNetwork load(File file) throws IOException, ClassNotFoundException {
		
		ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(new FileInputStream(file)));
		NeuralNetwork network = (NeuralNetwork) ois.readObject();
		network.w = (Matrix[]) ois.readObject();
		ois.close();
		
		return network;
		
	}
	
}
