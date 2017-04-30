package core;

import static java.lang.Math.cos;
import static java.lang.Math.exp;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

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
	private static final int SAMPLES = 2;
	
	public static void main(String[] args) {
		
		Random r = new Random();
		
		float[] xData = new float[2 * SAMPLES];
		float[] yData = new float[SAMPLES];
		
		double maxZ = 0;
		double minZ = 1;
		for (int sample = 0; sample < SAMPLES; sample++) {
			
			double x = r.nextDouble();
			double y = r.nextDouble();
			
			double x_ = 2 * x - 1;
			double y_ = 2 * y - 1;
			
			double z = sin(x_ * x_ / 2 - y_ * y_ / 4 + 3) * cos(2 * x_ + 1 - exp(y_));
			maxZ = max(z, maxZ);
			minZ = min(z, minZ);
			
			xData[sample] = (float) x_;
			xData[SAMPLES + sample] = (float) y_;
			yData[sample] = (float) z;
			
		}
		
		for (int sample = 0; sample < SAMPLES; sample++) {
			yData[sample] = (float) ((yData[sample] - minZ) / (maxZ - minZ));
		}
		Matrix x = new Matrix(2, SAMPLES, xData);
		Matrix y = new Matrix(1, SAMPLES, yData);
		
		// Matrix x = new Matrix(2, 1, new float[] { 0f, 0f, });
		// Matrix y = new Matrix(1, 1, new float[] { 1f });
		
		NeuralNetwork network = new NeuralNetwork(2, 1, 5, 2, r);
		
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
				
				perturbedWeights[i][j] = initialWeights[i][j] - EPSILON;
				network.getW()[i].setData(perturbedWeights[i]);
				float loss1 = network.getCost(x, y);
				
				djdw2Data[i][j] = (loss2 - loss1) / (2 * EPSILON);
				
				System.gc();
				
			}
			network.getW()[i].setData(initialWeights[i]);
		}
		
		Matrix[] djdw = network.getCostPrime(x, y);
		Matrix[] djdw2 = new Matrix[djdw.length];
		for (int i = 0; i < djdw2.length; i++) {
			Matrix.divide(djdw[i], SAMPLES, djdw[i]);
			djdw2[i] = new Matrix(djdw[i].getRows(), djdw[i].getColumns(), djdw2Data[i]);
		}
		
		for (int i = 0; i < initialWeights.length; i++) {
			Matrix sub = new Matrix(djdw2[i]);
			Matrix.sub(djdw[i], djdw2[i], sub);
			
			Matrix add = new Matrix(djdw2[i]);
			Matrix.add(djdw[i], djdw2[i], add);
			
			double subnorm = norm(sub);
			double addnorm = norm(add);
			System.out.println(subnorm / addnorm);
			
		}
		
	}
	
	public static double norm(Matrix a) {
		
		Matrix.pow(a, 2, a);
		return sqrt(Matrix.sum(a));
		
	}
	
}
