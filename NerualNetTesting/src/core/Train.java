package core;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

public class Train {
	
	private static final int SAMPLE_WIDTH = 28;
	private static final int SAMPLE_HEIGHT = 28;
	
	private static final String TRAIN_IMAGES = "train-images.idx3-ubyte";
	private static final String TRAIN_LABELS = "train-labels.idx1-ubyte";
	private static final int TRAIN_SAMPLES = 60000;
	
	private static final String TEST_IMAGES = "t10k-images.idx3-ubyte";
	private static final String TEST_LABELS = "t10k-labels.idx1-ubyte";
	private static final int TEST_SAMPLES = 10000;
	
	private static final int HIDDEN_LAYER_SIZE = 1000;
	private static final int HIDDEN_LAYER_AMOUNT = 3;
	
	private static final double LEARNING_RATE = 1;
	private static final double SAMPLE_PROPORTION = 0.002;
	
	private static long time = 0;
	
	public static void main(String[] args) {
		
		Matrix x = null;
		Matrix y = null;
		Matrix xTesting = null;
		Matrix yTesting = null;
		
		int trainSamples = (int) (TRAIN_SAMPLES * SAMPLE_PROPORTION);
		int testSamples = (int) (TEST_SAMPLES * SAMPLE_PROPORTION);
		
		try {
			
			x = getX(new FileInputStream(TRAIN_IMAGES), trainSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
			y = getY(new FileInputStream(TRAIN_LABELS), trainSamples);
			
			xTesting = getX(new FileInputStream(TEST_IMAGES), testSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
			yTesting = getY(new FileInputStream(TEST_LABELS), testSamples);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Random r = new Random(0);
		NN network = new NN(x.getColumns(), y.getColumns(), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_AMOUNT, r);
		
		long start = System.currentTimeMillis();
		
		for (int i = 0;; i++) {
			
			Matrix yHat = network.forward(x);
			
			network.findCostPrime(x, y, yHat);
			
			double cost = network.getCost(x, y, yHat);
			double testingCost = network.getCost(xTesting, yTesting, network.forward(xTesting));
			
			String gap = ",";
			System.out.println(i + gap + cost + gap + testingCost + gap + ((System.currentTimeMillis() - start) / 1000d));
			
			network.descend(LEARNING_RATE);
		}
		
	}
	
	private static Matrix getX(InputStream in, int samples, int width, int height) throws IOException {
		
		for (int i = 0; i < 16; i++) {
			in.read(); // metadata
		}
		double[][] data = new double[samples][SAMPLE_HEIGHT * SAMPLE_WIDTH];
		byte[] read = new byte[SAMPLE_HEIGHT * SAMPLE_WIDTH * samples];
		in.read(read, 0, SAMPLE_HEIGHT * SAMPLE_WIDTH * samples);
		for (int sample = 0; sample < samples; sample++) {
			for (int pixel = 0; pixel < SAMPLE_HEIGHT * SAMPLE_WIDTH; pixel++) {
				data[sample][pixel] = (read[sample * SAMPLE_HEIGHT * SAMPLE_WIDTH + pixel] & 0xff) / 255d;
			}
		}
		
		return new Matrix(samples, SAMPLE_HEIGHT * SAMPLE_WIDTH, data);
		
	}
	
	private static Matrix getY(InputStream in, int samples) throws IOException {
		
		for (int i = 0; i < 16; i++) {
			in.read(); // metadata
		}
		double[][] data = new double[samples][10];
		byte[] read = new byte[samples];
		in.read(read);
		for (int sample = 0; sample < samples; sample++) {
			data[sample][read[sample] & 0xff] = 1;
		}
		
		return new Matrix(samples, 10, data);
		
	}
	
	private static void time(String message) {
		
		if (time == 0 || message.equals("")) {
			time = System.currentTimeMillis();
		} else {
			long now = System.currentTimeMillis();
			System.out.println((now - time) / 1000d + "s elapsed " + message);
			time = now;
		}
		
	}
	
}