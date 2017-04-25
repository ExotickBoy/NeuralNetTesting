package trainers;

import static core.Matrix.multiply;
import static core.Matrix.sub;

import java.util.ArrayList;
import java.util.stream.IntStream;

import core.Matrix;
import core.NeuralNetwork;

public class GradientDescent extends OptimisationMethod {
	
	private float learningRate;
	
	public GradientDescent(float learningRate) {
		
		this.learningRate = learningRate;
		
	}
	
	@Override
	public void descend(NeuralNetwork network, float cost, ArrayList<Matrix> djdw, Matrix yHat) {
				
		ArrayList<Matrix> w = network.getW();
		
		IntStream.range(0, w.size()).parallel().forEach(i -> {
			
			w.set(i, sub(w.get(i), multiply(learningRate, djdw.get(i))));
			
		});
		
		network.setW(w);
		
	}
	
}
