package trainers;

import static core.Matrix.multiply;
import static core.Matrix.sub;

import java.util.stream.IntStream;

import core.Matrix;
import core.NeuralNetwork;

public class GradientDescent extends OptimisationMethod {
	
	private float learningRate;
	
	public GradientDescent(float learningRate) {
		
		this.learningRate = learningRate;
		
	}
	
	@Override
	public void descend(NeuralNetwork network, float cost, Matrix[] djdw, Matrix yHat) {
		
		Matrix[] w = network.getW();
		
		IntStream.range(0, w.length).parallel().forEach(i -> {
			
			w[i] = sub(w[i], multiply(learningRate, djdw[i]));
			
		});
		
	}
	
}
