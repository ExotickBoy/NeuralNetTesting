package trainers;

import core.Matrix;
import core.NeuralNetwork;

/**
 * 
 * A super class for all optimisation methods used by TrainingScheme
 *
 * @see TrainingScheme
 * 
 * @author Kacper
 *
 */
public abstract class OptimisationMethod {
	
	protected NeuralNetwork network;
	
	public OptimisationMethod(NeuralNetwork network) {
		
		this.network = network;
		
	}
	
	public abstract void descend(Matrix[] w, Matrix[] djdw);
	
}
