package trainers;

import java.util.ArrayList;

import core.Matrix;
import core.NeuralNetwork;

public abstract class TrainingScheme {

	private boolean defaultLimit = true;

	private boolean useMaxIterations = false;
	private boolean useMaxTime = true;
	private boolean useMinCost = false;
	private int maxIterations = 0;
	private int maxTime = 10 * 60;
	private int minCost = 0;

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

		System.out.println("Starting training");

		Matrix xTraining = getXTraining();
		Matrix yTraining = getYTraining();

		long startTime = System.currentTimeMillis();
		double timeElapsed = 0;

		Matrix yHat = network.forward(xTraining);
		double trainingCost = network.getCost(xTraining, yTraining, yHat);
		double testingCost;
		if (useTesting) {
			Matrix xTesting = getXTesting();
			Matrix yTesting = getYTesting();
			testingCost = network.getCost(xTesting, yTesting);
		} else {
			testingCost = 0;
		}
		
		if (callback != null) {
			callback.iterated(network, 0, trainingCost, testingCost, timeElapsed);
		}

		for (int iteration = 0; (!useMaxIterations || iteration < maxIterations)
				&& (!useMinCost || trainingCost < minCost) && (!useMaxTime || timeElapsed < maxTime); iteration++) {

			xTraining = getXTraining();
			yTraining = getYTraining();

			yHat = network.forward(xTraining);
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

		useTesting = true;
		this.allXTesting = xTesting;
		this.allYTesting = yTesting;

	}

	public void setMaxIterations(int maxIterations) {

		if (defaultLimit) {
			useMaxTime = false;
			defaultLimit = false;
		}

		useMaxIterations = true;
		this.maxIterations = maxIterations;

	}

	public void setMaxTime(int maxTime) {

		if (defaultLimit) {
			defaultLimit = false;
		}

		useMaxTime = true;
		this.maxTime = maxTime;

	}

	public void setMinCost(int minCost) {

		if (defaultLimit) {
			useMaxTime = false;
			defaultLimit = false;
		}

		useMinCost = true;
		this.minCost = minCost;

	}

	public interface CallBack {

		public void iterated(NeuralNetwork network, double iteration, double trainingCost, double testingCost,
				double timeElapsed);

	}

}
