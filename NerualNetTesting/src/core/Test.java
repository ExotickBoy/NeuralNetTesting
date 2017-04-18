
package core;

public class Test {
		
	public static void main(String[] args){
		
		Matrix a = new Matrix(3, 4, new float[]{
				1, 1, 1, 1,
				1, 1, 1, 1,
				1, 1, 1, 1
		});
		Matrix b = new Matrix(4, 2, new float[]{
				1, 1,
				1, 1,
				1, 1,
				1, 1
		});
		

		
		Matrix c0 = Matrix.dot(a, b);
		
		System.out.println(c0.toString());

	}
	
}