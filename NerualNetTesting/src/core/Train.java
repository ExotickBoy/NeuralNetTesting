package core;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
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

	private static final double LEARNING_RATE = .000005;
	private static final double SAMPLE_PROPORTION = 0.01;

	private static final double LOWER_COST_LIMIT = 0.01;

	private static final int STANDARD_ITERATION_LIMIT = 200;

	private static final int SGD_MINIBATCH_SIZE = 100;
	private static final int SGD_ITERATION_LIMIT = (int) (STANDARD_ITERATION_LIMIT * TRAIN_SAMPLES * SAMPLE_PROPORTION
			/ SGD_MINIBATCH_SIZE);

	private static long time = 0;

	public static void main(String[] args) {

		Matrix x = null;
		Matrix y = null;
		Matrix xTesting = null;
		Matrix yTesting = null;

		int trainSamples = (int) (TRAIN_SAMPLES * SAMPLE_PROPORTION);
		int testSamples = (int) (TEST_SAMPLES * SAMPLE_PROPORTION);

		boolean isStochastic = false;

		if (args.length == 1) {

			boolean isRecognisedArg = true;

			switch (args[0]) {

			case "-s":

				System.out.println("Using sgd");
				isStochastic = true;

				break;

			default:

				isRecognisedArg = false;

				break;

			}

			assert isRecognisedArg;

		}

		try {

			x = getX(new FileInputStream(TRAIN_IMAGES), trainSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
			y = getY(new FileInputStream(TRAIN_LABELS), trainSamples);

			xTesting = getX(new FileInputStream(TEST_IMAGES), testSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
			yTesting = getY(new FileInputStream(TEST_LABELS), testSamples);

		} catch (IOException e) {
			e.printStackTrace();
		}

		Random r = new Random(0);
		NN network = new NN(x.getColumns(), y.getColumns(), HIDDEN_LAYER_SIZE, r);

		long start = System.currentTimeMillis();

		double cost = Double.MAX_VALUE;

		if (!isStochastic) { // Standard gradient descent

			for (int i = 0; i < STANDARD_ITERATION_LIMIT && cost > LOWER_COST_LIMIT; i++) {

				Matrix yHat = network.forward(x);

				network.findCostPrime(x, y, yHat);

				cost = network.getCost(x, y, yHat);
				double testingCost = network.getCost(xTesting, yTesting, network.forward(xTesting));

				String gap = ",";
				System.out.println(
						i + gap + cost + gap + testingCost + gap + ((System.currentTimeMillis() - start) / 1000d));

				network.descend(LEARNING_RATE);
			}

		} else { // Stochastic gradient descent

			ArrayList<Matrix> miniBatchesX = new ArrayList<Matrix>();
			ArrayList<Matrix> miniBatchesY = new ArrayList<Matrix>();

			int numMinibatches = (int) (SAMPLE_PROPORTION * TRAIN_SAMPLES / SGD_MINIBATCH_SIZE);

			// Split x and y into minibatches
			for (int i = 0; i < numMinibatches; i++) {

				Matrix batchX = new Matrix(SGD_MINIBATCH_SIZE, SAMPLE_HEIGHT * SAMPLE_WIDTH);
				Matrix batchY = new Matrix(SGD_MINIBATCH_SIZE, 10);

				for (int row = 0; row < batchX.getRows(); row++) {
					for (int col = 0; col < batchX.getColumns(); col++) {
						batchX.set(row, col, x.get(row + i * SGD_MINIBATCH_SIZE, col));
					}
				}

				for (int row = 0; row < batchY.getRows(); row++) {
					for (int col = 0; col < batchY.getColumns(); col++) {
						batchY.set(row, col, y.get(row + i * SGD_MINIBATCH_SIZE, col));
					}
				}

				miniBatchesX.add(batchX);
				miniBatchesY.add(batchY);

			}

			// Each iteration, chose a minibatch at random and descend
			for (int i = 0; i < SGD_ITERATION_LIMIT && cost > LOWER_COST_LIMIT; i++) {

				int index = r.nextInt(numMinibatches);

				Matrix batchX = miniBatchesX.get(index);
				Matrix batchY = miniBatchesY.get(index);

				Matrix yHat = network.forward(batchX);

				network.findCostPrime(batchX, batchY, yHat);

				cost = network.getCost(batchX, batchY, yHat);

				double testingCost = network.getCost(xTesting, yTesting, network.forward(xTesting));

				String gap = ",";
				System.out.println(
						i + gap + cost + gap + testingCost + gap + ((System.currentTimeMillis() - start) / 1000d));

				network.descend(LEARNING_RATE);

			}

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