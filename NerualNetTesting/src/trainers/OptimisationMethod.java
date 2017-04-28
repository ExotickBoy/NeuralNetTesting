package trainers;

import core.Matrix;

public abstract class OptimisationMethod {
	
	public abstract void descend(Matrix[] w, Matrix[] djdw);
	
}
