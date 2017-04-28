package trainers;

import core.Matrix;

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
	
	public abstract void descend(Matrix[] w, Matrix[] djdw);
	
}
