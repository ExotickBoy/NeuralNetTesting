package trainers;

import java.util.ArrayList;
import java.util.Random;

import core.Matrix;
import core.NeuralNetwork;

/**
 * 
 * Training Method which only uses a small portion off all samples in each iteration
 * 
 * @author Jamie
 *
 */
public class StochasticTraining extends TrainingScheme {
	
	private static final int SGD_MINIBATCH_SIZE = 100;
	
	private int index;
	private Random random;
	
	private ArrayList<Matrix> miniBatchesY;
	private ArrayList<Matrix> miniBatchesX;
	
	public StochasticTraining(Matrix xTraining, Matrix yTraining, NeuralNetwork network, OptimisationMethod descentMethod, Random random) {
		
		super(xTraining, yTraining, network, descentMethod);
		this.random = random;
		
		miniBatchesX = new ArrayList<Matrix>();
		miniBatchesY = new ArrayList<Matrix>();
		
		int numMinibatches = (int) (xTraining.getRows() / SGD_MINIBATCH_SIZE);
		
		// Split x and y into minibatches
		for (int i = 0; i < numMinibatches; i++) {
			
			Matrix batchX = new Matrix(xTraining.getRows(), SGD_MINIBATCH_SIZE);
			Matrix batchY = new Matrix(10, SGD_MINIBATCH_SIZE);
			
			for (int row = 0; row < batchX.getRows(); row++) {
				for (int col = 0; col < batchX.getColumns(); col++) {
					batchX.set(row, col, xTraining.get(row, col + SGD_MINIBATCH_SIZE * i));
				}
			}
			
			for (int row = 0; row < batchY.getRows(); row++) {
				for (int col = 0; col < batchY.getColumns(); col++) {
					batchY.set(row, col, yTraining.get(row, col + SGD_MINIBATCH_SIZE * i));
				}
			}
			
			miniBatchesX.add(batchX);
			miniBatchesY.add(batchY);
			
		}
		
	}
	
	@Override
	protected Matrix getXTraining() {
		
		return miniBatchesX.get(index);
		
	}
	
	@Override
	protected Matrix getYTraining() {
		
		return miniBatchesY.get(index);
		
	}
	
	@Override
	protected Matrix getYTesting() {
		
		return getAllYTesting();
		
	}
	
	@Override
	protected Matrix getXTesting() {
		
		return getAllXTesting();
		
	}
	
	@Override
	protected void iterateData() {
		
		index = random.nextInt(miniBatchesX.size());
		
	}
	
}
