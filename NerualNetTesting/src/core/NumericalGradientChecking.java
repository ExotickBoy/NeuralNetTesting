package core;

import java.util.Random;

/**
 * 
 * This class implements numerical gradient checking, outputs the difference between the numerically
 * and analytically computed gradients
 * 
 * @author Kacper
 *
 */
public class NumericalGradientChecking {
	
	private static final float EPSILON = 1e-4f;
	
	public static void main(String[] args) {
		
		// Matrix x = new Matrix(2, 5, new float[] { .1f, .9f, .4f, .5f, .6f, .4f, .1f, .3f, .9f,
		// .1f, });
		// Matrix y = new Matrix(1, 5, new float[] { .1f, .4f, .3f, .5f, .9f });
		
		Matrix x = new Matrix(2, 1, new float[] { 0f, 0f, });
		Matrix y = new Matrix(1, 1, new float[] { 1f });
		
		Random r = new Random();
		NeuralNetwork network = new NeuralNetwork(2, 1, 5, 1, r);
		
		float[][] djdw2Data = new float[network.getW().length][];
		
		float[][] initialWeights = new float[network.getW().length][];
		float[][] perturbedWeights = new float[network.getW().length][];
		for (int i = 0; i < network.getW().length; i++) {
			initialWeights[i] = network.getW()[i].getData();
			perturbedWeights[i] = initialWeights[i].clone();
			djdw2Data[i] = new float[network.getW()[i].getSize()];
		}
		
		for (int i = 0; i < network.getW().length; i++) {
			for (int j = 0; j < network.getW()[i].getSize(); j++) {
				
				perturbedWeights[i][j] = initialWeights[i][j] + EPSILON;
				network.getW()[i].setData(perturbedWeights[i]);
				float loss2 = network.getCost(x, y);
				
				perturbedWeights[i][j] = initialWeights[i][j] + EPSILON;
				network.getW()[i].setData(perturbedWeights[i]);
				float loss1 = network.getCost(x, y);
				
				djdw2Data[i][j] = (loss2 - loss1) / (2 * EPSILON);
				
			}
			network.getW()[i].setData(initialWeights[i]);
		}
		
		Matrix[] djdw = network.getCostPrime(x, y);
		Matrix[] djdw2 = new Matrix[djdw.length];
		for (int i = 0; i < djdw2.length; i++) {
			djdw2[i] = new Matrix(djdw[i].getRows(), djdw[i].getColumns(), djdw2Data[i]);
		}
		
		for (int i = 0; i < initialWeights.length; i++) {
			Matrix sub = new Matrix(djdw[i]);
			Matrix.sub(djdw[i], djdw2[i], sub);
			System.out.println(sub);
		}
		
	}
	
}
