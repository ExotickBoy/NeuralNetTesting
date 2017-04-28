package trainers;

import static core.Matrix.multiply;
import static core.Matrix.sub;

import java.util.stream.IntStream;

import core.Matrix;

public class GradientDescent extends OptimisationMethod {
	
	private float learningRate;
	
	public GradientDescent(float learningRate) {
		
		this.learningRate = learningRate;
		
	}
	
	@Override
	public void descend(Matrix[] w, Matrix[] djdw) {
		
		IntStream.range(0, w.length).parallel().forEach(i -> {
			
			w[i] = sub(w[i], multiply(learningRate, djdw[i]));
			
		});
		
	}
	
}
