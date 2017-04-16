
package core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.zip.GZIPOutputStream;

import trainers.BatchTraining;
import trainers.SimpleGradientDescent;
import trainers.StochasticTraining;
import trainers.TrainingScheme;

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
	private static final double SAMPLE_PROPORTION = .1;
	
	public static void main(String[] args) {
		
		boolean isStochastic = false;
		
		if (args.length == 1) {
			
			for (int i = 0; i < args.length; i++) {
				
				switch (args[i]) {
				
				case "-s":
					
					System.out.println("Using sgd");
					isStochastic = true;
					
					break;
				
				default:
					
					assert false;
					break;
				
				}
				
			}
			
		}
		
		int trainSamples = (int) (TRAIN_SAMPLES * SAMPLE_PROPORTION);
		int testSamples = (int) (TEST_SAMPLES * SAMPLE_PROPORTION);
		
		Matrix xTraining = getX(new File(TRAIN_IMAGES), trainSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
		Matrix yTraining = getY(new File(TRAIN_LABELS), trainSamples);
		
		Matrix xTesting = getX(new File(TEST_IMAGES), testSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
		Matrix yTesting = getY(new File(TEST_LABELS), testSamples);
		
		Random random = new Random();
		
		NeuralNetwork network = new NeuralNetwork(xTraining.getColumns(), yTraining.getColumns(), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_AMOUNT, random);
		
		TrainingScheme trainer;
		
		if (isStochastic) {
			
			trainer = new StochasticTraining(xTraining, yTraining, network, new SimpleGradientDescent(LEARNING_RATE), random);
			
		} else {
			
			trainer = new BatchTraining(xTraining, yTraining, network, new SimpleGradientDescent(LEARNING_RATE));
			
		}
		trainer.setTestingData(xTesting, yTesting);
		trainer.setCallBack((n, iteration, trainingCost, testingCost, timeElapsed) -> {
			
			System.out.println(iteration + "," + trainingCost + "," + testingCost + "," + timeElapsed);
			try {
				ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(new File("network" + iteration + ".nwk"))));
				oos.writeObject(network);
				oos.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		});
		
		trainer.train();
		
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(new File("network.nwk"))));
			oos.writeObject(network);
			oos.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	private static Matrix getX(File file, int samples, int width, int height) {
		
		try {
			
			InputStream in = new FileInputStream(file);
			
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
			
			in.close();
			
			return new Matrix(samples, SAMPLE_HEIGHT * SAMPLE_WIDTH, data);
			
		} catch (IOException e) {
			
			e.printStackTrace();
			return null;
		}
		
	}
	
	private static Matrix getY(File file, int samples) {
		
		try {
			
			InputStream in = new FileInputStream(file);
			
			for (int i = 0; i < 16; i++) {
				in.read();
			} // metadata
			double[][] data = new double[samples][10];
			byte[] read = new byte[samples];
			in.read(read);
			for (int sample = 0; sample < samples; sample++) {
				data[sample][read[sample] & 0xff] = 1;
			}
			
			in.close();
			
			return new Matrix(samples, 10, data);
			
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		
	}
	
}