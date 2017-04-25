package trainers;

import java.util.ArrayList;

import core.Matrix;
import core.NeuralNetwork;

public abstract class OptimisationMethod {
	
	public abstract void descend(NeuralNetwork network, float cost, ArrayList<Matrix> djdw, Matrix yHat);
	
}
