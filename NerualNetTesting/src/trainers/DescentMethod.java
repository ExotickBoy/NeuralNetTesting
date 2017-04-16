package trainers;

import java.util.ArrayList;

import core.Matrix;
import core.NeuralNetwork;

public abstract class DescentMethod {
	
	public abstract void descend(NeuralNetwork network, ArrayList<Matrix> djdw);
	
}
