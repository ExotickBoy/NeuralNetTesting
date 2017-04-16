package trainers;

import java.util.ArrayList;

import core.Matrix;
import core.NeuralNetwork;

public abstract class TrainingScheme {
	
	private boolean defaultLimit = true;
	
	private boolean useMaxIterations = false;
	private boolean useMaxTime = false;
	private boolean useMinCost = false;
	private int maxIterations = 0;
	private double maxTime = 0;
	private double minCost = .1;
	
	private Matrix allXTraining;
	private Matrix allYTraining;
	
	private boolean useTesting;
	private Matrix allXTesting;
	private Matrix allYTesting;
	
	private NeuralNetwork network;
	private DescentMethod descentMethod;
	
	private CallBack callback;
	
	public TrainingScheme(Matrix xTraining, Matrix yTraining, NeuralNetwork network, DescentMethod descentMethod) {
		
		this.allXTraining = xTraining;
		this.allYTraining = yTraining;
		
		this.network = network;
		
		this.descentMethod = descentMethod;
		
	}
	
	public final void train() {
				
		long startTime = System.currentTimeMillis();
		double timeElapsed = 0;
		
		Matrix yHat = network.forward(allXTraining);
		double trainingCost = network.getCost(allXTraining, allYTraining, yHat);
		double testingCost = useTesting ? network.getCost(allXTesting, allYTesting, network.forward(allXTesting)) : 0;
		
		if (callback != null) {
			callback.iterated(network, 0, trainingCost, testingCost, timeElapsed);
		}
		
		
		for (int iteration = 0; (!useMaxIterations || iteration < maxIterations) && (!useMinCost || trainingCost < minCost) && (!useMaxTime || timeElapsed < maxTime); iteration++) {
			
			Matrix xTraining = getXTraining();
			Matrix yTraining = getYTraining();
			
			yHat = network.forward(allXTraining);
			ArrayList<Matrix> djdw = network.getCostPrime(xTraining, yTraining, yHat);
			
			descentMethod.descend(network, djdw);
			
			trainingCost = network.getCost(xTraining, yTraining, yHat);
			if (useTesting) {
				Matrix xTesting = getXTesting();
				Matrix yTesting = getYTesting();
				testingCost = network.getCost(xTesting, yTesting);
			} else {
				testingCost = 0;
			}
			
			iterateData();
			
			timeElapsed = (System.currentTimeMillis() - startTime) / 1000d;
			if (callback != null) {
				callback.iterated(network, iteration + 1, trainingCost, testingCost, timeElapsed);
			}
			
		}
		
		System.out.println("Trainging Completed");
		
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
	
	public void setMaxTime(double maxTime) {
		
		if (defaultLimit) {
			useMinCost = false;
			defaultLimit = false;
		}
		
		useMaxTime = true;
		this.maxTime = maxTime;
		
	}
	
	public void setMinCost(double minCost) {
		
		if (defaultLimit) {
			useMinCost = false;
			defaultLimit = false;
		}
		
		useMinCost = true;
		this.minCost = minCost;
		
	}
	
	public interface CallBack {
		
		public void iterated(NeuralNetwork network, int iteration, double trainingCost, double testingCost, double timeElapsed);
		
	}
	
}
