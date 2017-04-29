package trainers;

import core.Matrix;
import core.NeuralNetwork;

/**
 * Simplest linear method for finding the minimum
 * 
 * @author Kacper
 *
 */
public class GradientDescent extends OptimisationMethod {
	
	private float learningRate;
	
	public GradientDescent(NeuralNetwork network, float learningRate) {
		
		super(network);
		
		this.learningRate = learningRate;
		
	}
	
	@Override
	public void descend(Matrix[] w, Matrix[] djdw) {
		
		for (int i = 0; i < w.length; i++) {
			
			Matrix.multiply(learningRate, djdw[i], djdw[i]);
			Matrix.sub(w[i], djdw[i], w[i]);
			
		}
		
	}
	
}
