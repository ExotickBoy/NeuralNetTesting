
package core;

import static java.lang.Math.ceil;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import trainers.BatchTraining;
import trainers.GradientDescent;
import trainers.StochasticTraining;
import trainers.TrainingScheme;

/**
 * 
 * This class is an implementation of training a neural network on the MNIST data set
 * 
 * @author Kacper
 *
 */
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
	private static final int HIDDEN_LAYER_AMOUNT = 2;
	
	private static boolean isStochastic;
	private static boolean useTesting;
	private static double learningRate;
	private static double sampleProportion;
	private static boolean saveEachIteration;
	private static boolean willLoadLast;
	
	public static void main(String[] args) {
		
		isStochastic = false;
		useTesting = false;
		learningRate = 0.05;
		sampleProportion = .01;
		saveEachIteration = false;
		willLoadLast = false;
		
		for (int i = 0; i < args.length; i++) {
			
			switch (args[i]) {
			
			case "-s":
				
				isStochastic = true;
				break;
			
			case "-t":
				
				useTesting = true;
				break;
			
			case "-r":
				learningRate = Double.valueOf(args[++i]);
				break;
			
			case "-p":
				
				sampleProportion = Double.valueOf(args[++i]);
				break;
			
			case "-S":
				
				saveEachIteration = true;
				break;
			
			case "-l":
				
				willLoadLast = true;
				break;
			
			default:
				
				assert false;
				break;
			
			}
			
		}
		
		int trainSamples = (int) (ceil(TRAIN_SAMPLES * sampleProportion));
		int testSamples = (int) (ceil(TEST_SAMPLES * sampleProportion));
		
		Matrix xTraining = getX(new File(TRAIN_IMAGES), trainSamples);
		Matrix yTraining = getY(new File(TRAIN_LABELS), trainSamples);
		
		Matrix xTesting = getX(new File(TEST_IMAGES), testSamples);
		Matrix yTesting = getY(new File(TEST_LABELS), testSamples);
		
		Random random = new Random();
		
		NeuralNetwork network = null;
		if (willLoadLast) {
			
			ArrayList<File> files = new ArrayList<>(Arrays.asList(new File("nets/").listFiles()));
			File file = files.stream().max((a, b) -> {
				
				return a.getName().compareTo(b.getName());
				
			}).orElse(null);
			
			try {
				
				network = NeuralNetwork.load(file);
				
			} catch (ClassNotFoundException | IOException e) {
				
				e.printStackTrace();
				
			}
			
		}
		
		if (network == null) {
			
			network = new NeuralNetwork(xTraining.getRows(), yTraining.getRows(), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_AMOUNT, random);
			
		}
		
		TrainingScheme trainer;
		
		if (isStochastic) {
			
			trainer = new StochasticTraining(xTraining, yTraining, network, new GradientDescent(network, (float) learningRate), random);
			System.out.println("Using Stochastic Training");
			
		} else {
			
			trainer = new BatchTraining(xTraining, yTraining, network, new GradientDescent(network, (float) learningRate));
			System.out.println("Using Batch Training");
			
		}
		
		if (useTesting) {
			trainer.setTestingData(xTesting, yTesting);
			System.out.println("Using Testing");
		}
		trainer.setCallBack(Train::callback);
		
		trainer.train();
		
		try {
			
			network.save(new File("nets/network.nwk"));
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	/**
	 * 
	 * This is called from inside of TrainingScheme for every iteration
	 * 
	 */
	public static void callback(NeuralNetwork network, int iteration, double trainingCost, double testingCost, double timeElapsed) {
		
		System.out.println(iteration + "," + trainingCost + "," + testingCost + "," + timeElapsed);
		
		if (saveEachIteration) {
			
			try {
				
				network.save(new File("nets/network" + iteration + ".nwk"));
				
			} catch (IOException e) {
				e.printStackTrace();
			}
			
		}
		
	}
	
	/**
	 * 
	 * Loads the input MNIST data
	 * 
	 * @param file
	 *            - location of the file
	 * @param samples
	 *            - amount of samples that will be read
	 * @return the matrix where each column is a sample and the rows are the dimensions of the input
	 */
	private static Matrix getX(File file, int samples) {
		
		try {
			
			InputStream in = new FileInputStream(file);
			in.read(new byte[16]);
			
			float[] data = new float[SAMPLE_HEIGHT * SAMPLE_WIDTH * samples];
			
			byte[] read = new byte[SAMPLE_HEIGHT * SAMPLE_WIDTH * samples];
			in.read(read, 0, SAMPLE_HEIGHT * SAMPLE_WIDTH * samples);
			
			for (int pixel = 0; pixel < SAMPLE_HEIGHT * SAMPLE_WIDTH; pixel++) {
				for (int sample = 0; sample < samples; sample++) {
					
					data[pixel * SAMPLE_HEIGHT * SAMPLE_WIDTH + sample] = (float) ((read[sample] & 0xff) / 255f);
					
				}
				
			}
			
			in.close();
			
			return new Matrix(SAMPLE_HEIGHT * SAMPLE_WIDTH, samples, data);
			
		} catch (IOException e) {
			
			e.printStackTrace();
			return null;
		}
		
	}
	
	/**
	 * 
	 * Loads the output MNIST data
	 * 
	 * @param file
	 *            - location of the file
	 * @param samples
	 *            - amount of samples that will be read
	 * @return the matrix where each column is a sample and the rows are the dimensions of the
	 *         output
	 */
	private static Matrix getY(File file, int samples) {
		
		try {
			
			InputStream in = new FileInputStream(file);
			in.read(new byte[16]);
			
			float[] data = new float[10 * samples];
			byte[] read = new byte[samples];
			in.read(read);
			for (int sample = 0; sample < samples; sample++) {
				
				data[(read[sample] & 0xff) * samples + sample] = 1;
				
			}
			
			in.close();
			
			return new Matrix(10, samples, data);
			
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		
	}
	
}