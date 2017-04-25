package trainers;

import core.Matrix;
import core.NeuralNetwork;

public class BatchTraining extends TrainingScheme {
	
	public BatchTraining(Matrix xTraining, Matrix yTraining, NeuralNetwork network, OptimisationMethod descentMethod) {
		
		super(xTraining, yTraining, network, descentMethod);
		
	}
	
	@Override
	protected Matrix getXTraining() {
		return getAllXTraining();
	}
	
	@Override
	protected Matrix getYTraining() {
		return getAllYTraining();
	}
	
	@Override
	protected Matrix getXTesting() {
		return getAllXTesting();
	}
	
	@Override
	protected Matrix getYTesting() {
		return getAllYTesting();
	}
	
	@Override
	protected void iterateData() {}
	
}
