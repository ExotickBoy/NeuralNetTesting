package trainers;

import core.Matrix;
import core.NeuralNetwork;

/**
 * 
 * This class is the implementation which trains a neural network
 * 
 * @see NeuralNetwork
 * @see OptimisationMethod
 * 
 * @author Kacper
 *
 */
public abstract class TrainingScheme {
	
	private boolean defaultLimit = true;
	
	private boolean useMaxIterations = false;
	private boolean useMaxTime = false;
	private boolean useMinCost = true;
	private int maxIterations = 0;
	private int maxTime = 0;
	private double minCost = 0.001;
	
	private Matrix allXTraining;
	private Matrix allYTraining;
	
	private boolean useTesting;
	private Matrix allXTesting;
	private Matrix allYTesting;
	
	private NeuralNetwork network;
	private OptimisationMethod descentMethod;
	
	private CallBack callback;
	
	public TrainingScheme(Matrix xTraining, Matrix yTraining, NeuralNetwork network, OptimisationMethod descentMethod) {
		
		assert xTraining.getRows() == network.getInputLayerSize() && yTraining.getRows() == network.getOutputLayerSize();
		
		this.allXTraining = xTraining;
		this.allYTraining = yTraining;
		
		this.network = network;
		
		this.descentMethod = descentMethod;
		
	}
	
	/**
	 * 
	 * Begins the process of training the network, it will train until the finishing conditions are
	 * met, by default minCost = 0.001
	 * 
	 */
	public final void train() {
		
		System.out.println("Starting training");
		
		Matrix xTraining = getXTraining();
		Matrix yTraining = getYTraining();
		
		long startTime = System.currentTimeMillis();
		double timeElapsed = 0;
		
		double testingCost;
		if (useTesting) {
			Matrix xTesting = getXTesting();
			Matrix yTesting = getYTesting();
			testingCost = network.getCost(xTesting, yTesting);
		} else {
			testingCost = 0;
		}
		Matrix yHat = network.forward(xTraining);
		double trainingCost = network.getCost(xTraining, yTraining, yHat);
		
		if (callback != null) {
			callback.iterated(network, 0, trainingCost, testingCost, timeElapsed);
		}
		
		for (int iteration = 0; (!useMaxIterations || iteration < maxIterations) && (!useMinCost || trainingCost > minCost) && (!useMaxTime || timeElapsed < maxTime); iteration++) {
			
			xTraining = getXTraining();
			yTraining = getYTraining();
			
			Matrix[] djdw = network.getCostPrime(xTraining, yTraining, yHat);
			
			descentMethod.descend(network.getW(), djdw);
			
			if (useTesting) {
				Matrix xTesting = getXTesting();
				Matrix yTesting = getYTesting();
				testingCost = network.getCost(xTesting, yTesting);
			} else {
				testingCost = 0;
			}
			yHat = network.forward(xTraining);
			trainingCost = network.getCost(xTraining, yTraining, yHat);
			
			iterateData();
			
			timeElapsed = (System.currentTimeMillis() - startTime) / 1000d;
			if (callback != null) {
				callback.iterated(network, iteration + 1, trainingCost, testingCost, timeElapsed);
			}
			
			System.gc(); // to make sure that there is always as much free video memory as possible
			
		}
		
		System.out.println("Training Completed");
		
	}
	
	protected abstract Matrix getYTraining();
	
	protected abstract Matrix getXTraining();
	
	protected abstract Matrix getYTesting();
	
	protected abstract Matrix getXTesting();
	
	protected abstract void iterateData();
	
	protected Matrix getAllXTraining() {
		
		return allXTraining;
		
	}
	
	protected Matrix getAllYTraining() {
		
		return allYTraining;
		
	}
	
	protected Matrix getAllXTesting() {
		
		return allXTesting;
		
	}
	
	protected Matrix getAllYTesting() {
		
		return allYTesting;
		
	}
	
	protected boolean usesTesting() {
		
		return useTesting;
		
	}
	
	public void setCallBack(CallBack callback) {
		
		this.callback = callback;
		
	}
	
	public void setTestingData(Matrix xTesting, Matrix yTesting) {
		
		assert xTesting.getRows() == network.getInputLayerSize() && yTesting.getRows() == network.getOutputLayerSize();
		
		useTesting = true;
		this.allXTesting = xTesting;
		this.allYTesting = yTesting;
		
	}
	
	public void setMaxIterations(int maxIterations) {
		
		if (defaultLimit) {
			useMinCost = false;
			defaultLimit = false;
		}
		
		useMaxIterations = true;
		this.maxIterations = maxIterations;
		
	}
	
	public void setMaxTime(int maxTime) {
		
		if (defaultLimit) {
			useMinCost = false;
			defaultLimit = false;
		}
		
		useMaxTime = true;
		this.maxTime = maxTime;
		
	}
	
	public void setMinCost(double minCost) {
		
		if (defaultLimit) {
			useMinCost = true;
			defaultLimit = false;
		}
		
		useMinCost = true;
		this.minCost = minCost;
		
	}
	
	public interface CallBack {
		
		public void iterated(NeuralNetwork network, int iteration, double trainingCost, double testingCost, double timeElapsed);
		
	}
	
}
