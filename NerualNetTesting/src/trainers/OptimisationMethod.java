package trainers;

import core.Matrix;
import core.NeuralNetwork;

public abstract class OptimisationMethod {
	
	public abstract void descend(NeuralNetwork network, float cost, Matrix[] djdw, Matrix yHat);
	
}
