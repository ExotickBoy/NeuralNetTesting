
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
	private static final int HIDDEN_LAYER_AMOUNT = 5;
	
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
		
		Matrix xTraining = getX(new File(TRAIN_IMAGES), trainSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
		Matrix yTraining = getY(new File(TRAIN_LABELS), trainSamples);
		
		Matrix xTesting = getX(new File(TEST_IMAGES), testSamples, SAMPLE_WIDTH, SAMPLE_HEIGHT);
		Matrix yTesting = getY(new File(TEST_LABELS), testSamples);
		
		Random random = new Random();
		
		NeuralNetwork network = null;
		if (willLoadLast) {
			
			ArrayList<File> files = new ArrayList<>(Arrays.asList(new File("nets/").listFiles()));
			File file = files.stream().max((a, b) -> {
				
				return a.getName().compareTo(b.getName());
				
			}).orElse(null);
			
			network = NeuralNetwork.load(file);
			
		}
		
		if (network == null) {
			
			network = new NeuralNetwork(xTraining.getRows(), yTraining.getRows(), HIDDEN_LAYER_SIZE, HIDDEN_LAYER_AMOUNT, 0, random);
			
		}
		
		TrainingScheme trainer;
		
		if (isStochastic) {
			
			trainer = new StochasticTraining(xTraining, yTraining, network, new GradientDescent((float) learningRate), random);
			System.out.println("Using Stochastic Training");
			
		} else {
			
			trainer = new BatchTraining(xTraining, yTraining, network, new GradientDescent((float) learningRate));
			System.out.println("Using Batch Training");
			
		}
		
		if (useTesting) {
			trainer.setTestingData(xTesting, yTesting);
			System.out.println("Using Testing");
		}
		trainer.setCallBack(Train::callback);
		
		trainer.train();
		network.save(new File("nets/network.nwk"));
		
	}
	
	public static void callback(NeuralNetwork network, int iteration, double trainingCost, double testingCost, double timeElapsed) {
		
		System.out.println(iteration + "," + trainingCost + "," + testingCost + "," + timeElapsed);
		
		if (saveEachIteration) {
			
			network.save(new File("nets/network" + iteration + ".nwk"));
			
		}
		
	}
	
	private static Matrix getX(File file, int samples, int width, int height) {
		
		try {
			
			InputStream in = new FileInputStream(file);
			
			for (int i = 0; i < 16; i++) {
				in.read(); // metadata
			}
			Matrix data = new Matrix(SAMPLE_HEIGHT * SAMPLE_WIDTH, samples);
			
			byte[] read = new byte[SAMPLE_HEIGHT * SAMPLE_WIDTH * samples];
			in.read(read, 0, SAMPLE_HEIGHT * SAMPLE_WIDTH * samples);
			
			for (int sample = 0; sample < samples; sample++) {
				for (int pixel = 0; pixel < SAMPLE_HEIGHT * SAMPLE_WIDTH; pixel++) {
					
					data.set(pixel, sample, (read[sample] & 0xff) / 255d);
					
				}
				
			}
			
			in.close();
			
			return data;
			
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
			Matrix data = new Matrix(10, samples);
			byte[] read = new byte[samples];
			in.read(read);
			for (int sample = 0; sample < samples; sample++) {
				data.set(read[sample] & 0xff, sample, 1);
			}
			
			in.close();
			
			return data;
			
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		
	}
	
}