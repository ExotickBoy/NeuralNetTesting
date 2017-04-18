
package core;

public class Test {
		
	public static void main(String[] args){
		
		Matrix a = new Matrix(2, 3, new float[]{
				1, 1, 1,
				1, 2, 1
		});
		Matrix b = new Matrix(4, 2, new float[]{
				1, 1,
				1, 1,
				1, 1,
				1, 1
		});
		

		
		Matrix c0 = Matrix.sigmoid(a);
		
		System.out.println(c0.toString());

	}
	
}